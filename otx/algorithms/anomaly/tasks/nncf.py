"""Anomaly Classification Task."""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import io
import json
import os
import re
from typing import Dict, Optional

import torch
from anomalib.models import AnomalyModule, get_model
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.nncf.callback import NNCFCallback
from anomalib.utils.callbacks.nncf.utils import (
    compose_nncf_config,
    is_state_nncf,
    wrap_nncf_model,
)
from pytorch_lightning import Trainer

from otx.algorithms.anomaly.adapters.anomalib.callbacks import ProgressCallback
from otx.algorithms.anomaly.adapters.anomalib.data import OTXAnomalyDataModule
from otx.algorithms.anomaly.adapters.anomalib.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)

from .inference import InferenceTask

logger = get_logger(__name__)


class NNCFTask(InferenceTask, IOptimizationTask):
    """Base Anomaly Task."""

    def __init__(self, task_environment: TaskEnvironment, **kwargs) -> None:
        """Task for compressing models using NNCF.

        Args:
            task_environment (TaskEnvironment): OTX Task environment.
        """
        self.compression_ctrl = None
        self.nncf_preset = "nncf_quantization"
        super().__init__(task_environment, **kwargs)
        self.optimization_type = ModelOptimizationType.NNCF

    def _set_attributes_by_hyperparams(self):
        quantization = self.hyper_parameters.nncf_optimization.enable_quantization
        pruning = self.hyper_parameters.nncf_optimization.enable_pruning
        if quantization and pruning:
            self.nncf_preset = "nncf_quantization_pruning"
            self.optimization_methods = [
                OptimizationMethod.QUANTIZATION,
                OptimizationMethod.FILTER_PRUNING,
            ]
            self.precision = [ModelPrecision.INT8]
            return
        if quantization and not pruning:
            self.nncf_preset = "nncf_quantization"
            self.optimization_methods = [OptimizationMethod.QUANTIZATION]
            self.precision = [ModelPrecision.INT8]
            return
        if not quantization and pruning:
            self.nncf_preset = "nncf_pruning"
            self.optimization_methods = [OptimizationMethod.FILTER_PRUNING]
            self.precision = [ModelPrecision.FP32]
            return
        raise RuntimeError("Not selected optimization algorithm")

    def load_model(self, otx_model: Optional[ModelEntity]) -> AnomalyModule:
        """Create and Load Anomalib Module from OTX Model.

        This method checks if the task environment has a saved OTX Model,
        and creates one. If the OTX model already exists, it returns the
        the model with the saved weights.

        Args:
            otx_model (Optional[ModelEntity]): OTX Model from the
                task environment.

        Returns:
            AnomalyModule: Anomalib
                classification or segmentation model with/without weights.
        """
        nncf_config_path = os.path.join(self.base_dir, "compression_config.json")

        with open(nncf_config_path, encoding="utf8") as nncf_config_file:
            common_nncf_config = json.load(nncf_config_file)

        self._set_attributes_by_hyperparams()
        self.optimization_config = compose_nncf_config(common_nncf_config, [self.nncf_preset])
        self.config.merge_with(self.optimization_config)
        model = get_model(config=self.config)
        if otx_model is None:
            raise ValueError("No trained model in project. NNCF require pretrained weights to compress the model")

        buffer = io.BytesIO(otx_model.get_data("weights.pth"))  # type: ignore
        model_data = torch.load(buffer, map_location=torch.device("cpu"))

        if is_state_nncf(model_data):
            logger.info("Loaded model weights from Task Environment and wrapped by NNCF")

            # Fix name mismatch for wrapped model by pytorch_lighting
            nncf_modules = {}
            pl_modules = {}
            for key in model_data["model"].keys():
                if key.startswith("model."):
                    new_key = key.replace("model.", "")
                    res = re.search(r"nncf_module\.(\w+)_feature_extractor\.(.*)", new_key)
                    if res:
                        new_key = f"nncf_module.{res.group(1)}_model.feature_extractor.{res.group(2)}"
                    nncf_modules[new_key] = model_data["model"][key]
                else:
                    pl_modules[key] = model_data["model"][key]
            model_data["model"] = nncf_modules

            self.compression_ctrl, model.model = wrap_nncf_model(
                model.model,
                self.optimization_config["nncf_config"],
                init_state_dict=model_data,
            )
            # Load extra parameters of pytorch_lighting model
            model.load_state_dict(pl_modules, strict=False)
        else:
            try:
                model.load_state_dict(model_data["model"])
                logger.info("Loaded model weights from Task Environment")
            except BaseException as exception:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") from exception

        return model

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        """Train the anomaly classification model.

        Args:
            optimization_type (OptimizationType): Type of optimization.
            dataset (DatasetEntity): Input dataset.
            output_model (ModelEntity): Output model to save the model weights.
            optimization_parameters (OptimizationParameters): Training parameters
        """
        logger.info("Optimization the model.")

        if optimization_type is not OptimizationType.NNCF:
            raise RuntimeError("NNCF is the only supported optimization")

        datamodule = OTXAnomalyDataModule(config=self.config, dataset=dataset, task_type=self.task_type)
        nncf_callback = NNCFCallback(config=self.optimization_config["nncf_config"])
        metrics_configuration = MetricsConfigurationCallback(
            task=self.config.dataset.task,
            image_metrics=self.config.metrics.image,
            pixel_metrics=self.config.metrics.get("pixel"),
        )
        post_processing_configuration = PostProcessingConfigurationCallback(
            normalization_method=NormalizationMethod.MIN_MAX,
            threshold_method=ThresholdMethod.ADAPTIVE,
            manual_image_threshold=self.config.metrics.threshold.manual_image,
            manual_pixel_threshold=self.config.metrics.threshold.manual_pixel,
        )
        callbacks = [
            ProgressCallback(parameters=optimization_parameters),
            MinMaxNormalizationCallback(),
            nncf_callback,
            metrics_configuration,
            post_processing_configuration,
        ]

        self.trainer = Trainer(**self.config.trainer, logger=False, callbacks=callbacks)
        self.trainer.fit(model=self.model, datamodule=datamodule)
        self.compression_ctrl = nncf_callback.nncf_ctrl
        output_model.model_format = ModelFormat.BASE_FRAMEWORK
        output_model.optimization_type = ModelOptimizationType.NNCF
        self.save_model(output_model)

        logger.info("Training completed.")

    def model_info(self) -> Dict:
        """Return model info to save the model weights.

        Returns:
           Dict: Model info.
        """
        return {
            "compression_state": self.compression_ctrl.get_compression_state(),  # type: ignore
            "meta": {
                "config": self.config,
                "nncf_enable_compression": True,
            },
            "model": self.model.state_dict(),
            "config": self.get_config(),
            "VERSION": 1,
        }

    def _export_to_onnx(self, onnx_path: str):
        """Export model to ONNX.

        Args:
             onnx_path (str): path to save ONNX file
        """
        self.compression_ctrl.export_model(onnx_path, "onnx_11")  # type: ignore
