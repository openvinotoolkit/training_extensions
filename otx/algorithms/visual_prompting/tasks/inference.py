"""Visual Prompting Task."""

# Copyright (C) 2023 Intel Corporation
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

import ctypes
import io
import time
import os
import shutil
import subprocess  # nosec
import tempfile
from glob import glob
from typing import Dict, List, Optional, Union
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything import sam_model_registry

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer

from otx.algorithms.anomaly.adapters.anomalib.callbacks import ProgressCallback
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.config import get_visual_promtping_config
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.visual_prompting.configs.base.configuration import VisualPromptingConfig
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.metrics import NullPerformance, Performance, ScoreMetric
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from pytorch_lightning import LightningModule
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets import OTXVisualPromptingDataModule

logger = get_logger()


# pylint: disable=too-many-instance-attributes
class InferenceTask(IInferenceTask, IEvaluationTask, IExportTask, IUnload):
    """Base Visual Prompting Task."""

    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None) -> None:
        """Train, Infer, Export, Optimize and Deploy an Visual Prompting Task.

        Args:
            task_environment (TaskEnvironment): OTX Task environment.
            output_path (Optional[str]): output path where task output are saved.
        """
        torch.backends.cudnn.enabled = True
        logger.info("Initializing the task environment.")
        self.task_environment = task_environment
        self.task_type = task_environment.model_template.task_type
        self.model_name = task_environment.model_template.name
        self.labels = task_environment.get_labels()

        template_file_path = task_environment.model_template.model_template_path
        self.base_dir = os.path.abspath(os.path.dirname(template_file_path))

        # Hyperparameters.
        self._work_dir_is_temp = False
        self.output_path = output_path
        if self.output_path is None:
            self.output_path = tempfile.mkdtemp(prefix="otx-visual_prompting")
            self._work_dir_is_temp = True
        self.config = self.get_config()

        # Set default model attributes.
        self.optimization_methods: List[OptimizationMethod] = []
        self.precision = [ModelPrecision.FP32]
        self.optimization_type = ModelOptimizationType.MO

        self.model = self.load_model(otx_model=task_environment.model)

        self.trainer: Trainer
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def get_config(self) -> Union[DictConfig, ListConfig]:
        """Get Visual Prompting Config from task environment.

        Returns:
            Union[DictConfig, ListConfig]: Visual Prompting config.
        """
        self.hyper_parameters: VisualPromptingConfig = self.task_environment.get_hyper_parameters()
        config = get_visual_promtping_config(task_name=self.model_name, otx_config=self.hyper_parameters)

        config.dataset.task = "visual_prompting"

        return config

    def load_model(self, otx_model: Optional[ModelEntity]) -> LightningModule:
        """Create and Load Visual Prompting Module.

        Currently, load model through `sam_model_registry` because there is only SAM.
        If other visual prompting model is added, loading model process must be changed.

        Args:
            otx_model (Optional[ModelEntity]): OTX Model from the task environment.

        Returns:
            LightningModule: Visual prompting model with/without weights.
        """
        if otx_model is None:
            backbone = self.config.model.backbone
        else:
            backbone = otx_model

        # TODO (sungchul): where can load_from be applied?
        checkpoint = self.config.model.checkpoint
        
        # TODO (sungchul): load model in different ways
        model = sam_model_registry[backbone](checkpoint=checkpoint)
        return model

    def cancel_training(self) -> None:
        """Cancel the training `after_batch_end`.

        This terminates the training; however validation is still performed.
        """
        logger.info("Cancel training requested.")
        self.trainer.should_stop = True

        # The runner periodically checks `.stop_training` file to ensure if cancellation is requested.
        cancel_training_file_path = os.path.join(self.config.project.path, ".stop_training")
        with open(file=cancel_training_file_path, mode="a", encoding="utf-8"):
            pass

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        """Perform inference on a dataset.

        Args:
            dataset (DatasetEntity): Dataset to infer.
            inference_parameters (InferenceParameters): Inference parameters.

        Returns:
            DatasetEntity: Output dataset with predictions.
        """
        logger.info("Performing inference on the validation set using the base torch model.")
        datamodule = OTXVisualPromptingDataModule(config=self.config, dataset=dataset)

        logger.info("Inference Configs '%s'", self.config)

        # Callbacks.
        callbacks = [ProgressCallback(parameters=inference_parameters)]

        self.trainer = Trainer(**self.config.trainer, logger=False, callbacks=callbacks)
        self.trainer.predict(model=self.model, datamodule=datamodule)
        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None) -> None:
        """Evaluate the performance on a result set.

        Args:
            output_resultset (ResultSetEntity): Result Set from which the performance is evaluated.
            evaluation_metric (Optional[str], optional): Evaluation metric. Defaults to None. Instead,
                metric is chosen depending on the task type.
        """
        metric: IPerformanceProvider
        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            metric = MetricsHelper.compute_f_measure(output_resultset)
        elif self.task_type == TaskType.ANOMALY_DETECTION:
            metric = MetricsHelper.compute_anomaly_detection_scores(output_resultset)
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            metric = MetricsHelper.compute_anomaly_segmentation_scores(output_resultset)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        output_resultset.performance = metric.get_performance()

        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            accuracy = MetricsHelper.compute_accuracy(output_resultset).get_performance()
            output_resultset.performance.dashboard_metrics.extend(accuracy.dashboard_metrics)

    def _export_to_onnx(self, onnx_path: str):
        """Export model to ONNX.

        Args:
             onnx_path (str): path to save ONNX file
        """
        height, width = self.config.model.input_size
        torch.onnx.export(
            model=self.model.model,
            args=torch.zeros((1, 3, height, width)).to(self.model.device),
            f=onnx_path,
            opset_version=11,
        )

    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = False,
    ) -> None:
        """Export model to OpenVINO IR.

        Args:
            export_type (ExportType): Export type should be ExportType.OPENVINO
            output_model (ModelEntity): The model entity in which to write the OpenVINO IR data
            precision (bool): Output model weights and inference precision
            dump_features (bool): Flag to return "feature_vector" and "saliency_map".

        Raises:
            Exception: If export_type is not ExportType.OPENVINO
        """
        if dump_features:
            logger.warning(
                "Feature dumping is not implemented for the anomaly task."
                "The saliency maps and representation vector outputs will not be dumped in the exported model."
            )

        if export_type == ExportType.ONNX:
            output_model.model_format = ModelFormat.ONNX
            output_model.optimization_type = ModelOptimizationType.ONNX
            if precision == ModelPrecision.FP16:
                raise RuntimeError("Export to FP16 ONNX is not supported")
        elif export_type == ExportType.OPENVINO:
            output_model.model_format = ModelFormat.OPENVINO
            output_model.optimization_type = ModelOptimizationType.MO
        else:
            raise RuntimeError(f"not supported export type {export_type}")

        self.precision[0] = precision
        output_model.has_xai = dump_features

        # pylint: disable=no-member; need to refactor this
        logger.info("Exporting the OpenVINO model.")
        onnx_path = os.path.join(self.config.project.path, "onnx_model.onnx")
        self._export_to_onnx(onnx_path)

        if export_type == ExportType.ONNX:
            with open(onnx_path, "rb") as file:
                output_model.set_data("model.onnx", file.read())
        else:
            optimize_command = ["mo", "--input_model", onnx_path, "--output_dir", self.config.project.path]
            if precision == ModelPrecision.FP16:
                optimize_command.append("--compress_to_fp16")
            subprocess.run(optimize_command, check=True)
            bin_file = glob(os.path.join(self.config.project.path, "*.bin"))[0]
            xml_file = glob(os.path.join(self.config.project.path, "*.xml"))[0]
            with open(bin_file, "rb") as file:
                output_model.set_data("openvino.bin", file.read())
            with open(xml_file, "rb") as file:
                output_model.set_data("openvino.xml", file.read())

        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods

        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))
        self._set_metadata(output_model)

    def model_info(self) -> Dict:
        """Return model info to save the model weights.

        Returns:
           Dict: Model info.
        """
        return {
            "model": self.model.state_dict(),
            "config": self.get_config(),
            "version": self.trainer.logger.version,
        }

    def save_model(self, output_model: ModelEntity) -> None:
        """Save the model after training is completed.

        Args:
            output_model (ModelEntity): Output model onto which the weights are saved.
        """
        logger.info("Saving the model weights.")
        model_info = self.model_info()
        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))
        self._set_metadata(output_model)

        if hasattr(self.model, "image_metrics"):
            f1_score = self.model.image_metrics.F1Score.compute().item()
            output_model.performance = Performance(score=ScoreMetric(name="F1 Score", value=f1_score))
        else:
            output_model.performance = NullPerformance()
        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods

    def _set_metadata(self, output_model: ModelEntity):
        if hasattr(self.model, "image_threshold"):
            output_model.set_data("image_threshold", self.model.image_threshold.value.cpu().numpy().tobytes())
        if hasattr(self.model, "pixel_threshold"):
            output_model.set_data("pixel_threshold", self.model.pixel_threshold.value.cpu().numpy().tobytes())
        if hasattr(self.model, "normalization_metrics"):
            output_model.set_data("min", self.model.normalization_metrics.state_dict()["min"].cpu().numpy().tobytes())
            output_model.set_data("max", self.model.normalization_metrics.state_dict()["max"].cpu().numpy().tobytes())
        else:
            logger.warning(
                "The model was not trained before saving. This will lead to incorrect normalization of the heatmaps."
            )

    @staticmethod
    def _is_docker() -> bool:
        """Check whether the task runs in docker container.

        Returns:
            bool: True if task runs in docker, False otherwise.
        """
        path = "/proc/self/cgroup"
        is_in_docker = False
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as file:
                is_in_docker = is_in_docker or any("docker" in line for line in file)
        is_in_docker = is_in_docker or os.path.exists("/.dockerenv")
        return is_in_docker

    def unload(self) -> None:
        """Unload the task."""
        self.cleanup()

        if self._is_docker():
            logger.warning("Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            ctypes.string_at(0)

        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(
                "Done unloading. Torch is still occupying %f bytes of GPU memory",
                torch.cuda.memory_allocated(),
            )

    def cleanup(self) -> None:
        """Clean up work directory."""
        if self._work_dir_is_temp and os.path.exists(self.config.project.path):
            shutil.rmtree(self.config.project.path, ignore_errors=False)
