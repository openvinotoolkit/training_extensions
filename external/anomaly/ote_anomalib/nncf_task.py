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
import subprocess  # nosec
from glob import glob
from typing import Dict, Optional

import torch
from anomalib.models import AnomalyModule, get_model
from anomalib.utils.callbacks import MinMaxNormalizationCallback
from anomalib.utils.callbacks.nncf.callback import NNCFCallback
from anomalib.utils.callbacks.nncf.utils import (
    compose_nncf_config,
    is_state_nncf,
    wrap_nncf_model,
)
from ote_anomalib import AnomalyInferenceTask
from ote_anomalib.callbacks import ProgressCallback
from ote_anomalib.data import OTEAnomalyDataModule
from ote_anomalib.logging import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.metrics import Performance, ScoreMetric
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from pytorch_lightning import Trainer

logger = get_logger(__name__)


class AnomalyNNCFTask(AnomalyInferenceTask, IOptimizationTask):
    """Base Anomaly Task."""

    def __init__(self, task_environment: TaskEnvironment) -> None:
        """Task for compressing models using NNCF.

        Args:
            task_environment (TaskEnvironment): OTE Task environment.
        """
        self.val_dataloader = None
        self.compression_ctrl = None
        self.nncf_preset = "nncf_quantization"
        super().__init__(task_environment)
        self.optimization_type = ModelOptimizationType.NNCF

    def _set_attributes_by_hyperparams(self):
        quantization = self.hyper_parameters.nncf_optimization.enable_quantization
        pruning = self.hyper_parameters.nncf_optimization.enable_pruning
        if quantization and pruning:
            self.nncf_preset = "nncf_quantization_pruning"
            self.optimization_methods = [OptimizationMethod.QUANTIZATION, OptimizationMethod.FILTER_PRUNING]
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

    def load_model(self, ote_model: Optional[ModelEntity]) -> AnomalyModule:
        """Create and Load Anomalib Module from OTE Model.

        This method checks if the task environment has a saved OTE Model,
        and creates one. If the OTE model already exists, it returns the
        the model with the saved weights.

        Args:
            ote_model (Optional[ModelEntity]): OTE Model from the
                task environment.

        Returns:
            AnomalyModule: Anomalib
                classification or segmentation model with/without weights.
        """
        nncf_config_path = os.path.join(self.base_dir, "compression_config.json")

        with open(nncf_config_path) as nncf_config_file:
            common_nncf_config = json.load(nncf_config_file)

        self._set_attributes_by_hyperparams()
        self.optimization_config = compose_nncf_config(common_nncf_config, [self.nncf_preset])
        self.config.merge_with(self.optimization_config)
        model = get_model(config=self.config)
        if ote_model is None:
            raise ValueError("No trained model in project. NNCF require pretrained weights to compress the model")
        else:
            buffer = io.BytesIO(ote_model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            if is_state_nncf(model_data):
                logger.info("Loaded model weights from Task Environment and wrapped by NNCF")

                # Workaround to fix incorrect loading state for wrapped pytorch_lighting model
                new_model = dict()
                for key in model_data["model"].keys():
                    if key.startswith("model."):
                        new_model[key.replace("model.", "")] = model_data["model"][key]
                model_data["model"] = new_model

                self.compression_ctrl, model.model = wrap_nncf_model(
                    model.model, self.optimization_config["nncf_config"], init_state_dict=model_data
                )
            else:
                try:
                    model.load_state_dict(model_data["model"])
                    logger.info("Loaded model weights from Task Environment")
                except BaseException as exception:
                    raise ValueError(
                        "Could not load the saved model. The model file structure is invalid."
                    ) from exception

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

        # config = self.get_config()
        # logger.info("Training Configs '%s'", config)

        datamodule = OTEAnomalyDataModule(config=self.config, dataset=dataset, task_type=self.task_type)
        # Setup dataset to initialization of compressed model
        # datamodule.setup(stage="fit")
        # nncf_config = yaml.safe_load(OmegaConf.to_yaml(self.config['nncf_config']))

        nncf_callback = NNCFCallback(nncf_config=self.optimization_config["nncf_config"])
        callbacks = [
            ProgressCallback(parameters=optimization_parameters),
            MinMaxNormalizationCallback(),
            nncf_callback,
        ]

        self.trainer = Trainer(**self.config.trainer, logger=False, callbacks=callbacks)
        self.trainer.fit(model=self.model, datamodule=datamodule)
        self.compression_ctrl = nncf_callback.nncf_ctrl
        self.save_model(output_model)

        logger.info("Training completed.")

    def _model_info(self) -> Dict:
        """Return model info to save the model weights.

        Returns:
           Dict: Model info.
        """

        return {
            "compression_state": self.compression_ctrl.get_compression_state(),
            "meta": {
                "config": self.config,
                "nncf_enable_compression": True,
            },
            "model": self.model.state_dict(),
            "config": self.get_config(),
            "VERSION": 1,
        }

    def save_model(self, output_model: ModelEntity) -> None:
        """Save the model after training is completed.

        Args:
            output_model (ModelEntity): Output model onto which the weights are saved.
        """
        logger.info("Saving the model weights.")
        model_info = self._model_info()
        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))
        self._set_metadata(output_model)

        f1_score = self.model.image_metrics.F1.compute().item()
        output_model.performance = Performance(score=ScoreMetric(name="F1 Score", value=f1_score))
        output_model.precision = self.precision

    def _export_to_onnx(self, onnx_path: str):
        """Export model to ONNX

        Args:
             onnx_path (str): path to save ONNX file
        """
        self.compression_ctrl.export_model(onnx_path, "onnx_11")
