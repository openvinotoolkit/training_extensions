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

import ctypes
import io
import os
import shutil
import subprocess  # nosec
import tempfile
from glob import glob
from typing import Dict, List, Optional, Union

import torch
from anomalib.models import AnomalyModule, get_model
from anomalib.utils.callbacks import MinMaxNormalizationCallback
from omegaconf import DictConfig, ListConfig
from ote_anomalib.callbacks import AnomalyInferenceCallback, ProgressCallback
from ote_anomalib.configs import get_anomalib_config
from ote_anomalib.data import OTEAnomalyDataModule
from ote_anomalib.logging import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.metrics import Performance, ScoreMetric
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from ote_sdk.entities.model_template import TaskType
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from pytorch_lightning import Trainer

logger = get_logger(__name__)


# pylint: disable=too-many-instance-attributes
class AnomalyInferenceTask(IInferenceTask, IEvaluationTask, IExportTask, IUnload):
    """Base Anomaly Task."""

    def __init__(self, task_environment: TaskEnvironment) -> None:
        """Train, Infer, Export, Optimize and Deploy an Anomaly Classification Task.

        Args:
            task_environment (TaskEnvironment): OTE Task environment.
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
        self.project_path: str = tempfile.mkdtemp(prefix="ote-anomalib")
        self.config = self.get_config()

        # Set default model attributes.
        self.optimization_methods: List[OptimizationMethod] = []
        self.precision = [ModelPrecision.FP32]
        self.optimization_type = ModelOptimizationType.MO

        self.model = self.load_model(ote_model=task_environment.model)

        self.trainer: Trainer

    def get_config(self) -> Union[DictConfig, ListConfig]:
        """Get Anomalib Config from task environment.

        Returns:
            Union[DictConfig, ListConfig]: Anomalib config.
        """
        self.hyper_parameters = self.task_environment.get_hyper_parameters()
        config = get_anomalib_config(task_name=self.model_name, ote_config=self.hyper_parameters)
        config.project.path = self.project_path

        config.dataset.task = "classification"

        return config

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
        model = get_model(config=self.config)
        if ote_model is None:
            logger.info(
                "No trained model in project yet. Created new model with '%s'",
                self.model_name,
            )
        else:
            buffer = io.BytesIO(ote_model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            try:
                model.load_state_dict(model_data["model"])
                logger.info("Loaded model weights from Task Environment")

            except BaseException as exception:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") from exception

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
        config = self.get_config()
        datamodule = OTEAnomalyDataModule(config=config, dataset=dataset, task_type=self.task_type)

        logger.info("Inference Configs '%s'", config)

        # Callbacks.
        progress = ProgressCallback(parameters=inference_parameters)
        inference = AnomalyInferenceCallback(dataset, self.labels, self.task_type)
        normalize = MinMaxNormalizationCallback()
        callbacks = [progress, normalize, inference]

        self.trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks)
        self.trainer.predict(model=self.model, datamodule=datamodule)
        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None) -> None:
        """Evaluate the performance on a result set.

        Args:
            output_resultset (ResultSetEntity): Result Set from which the performance is evaluated.
            evaluation_metric (Optional[str], optional): Evaluation metric. Defaults to None. Instead,
                metric is chosen depending on the task type.
        """
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
        """Export model to ONNX

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

    def export(self, export_type: ExportType, output_model: ModelEntity) -> None:
        """Export model to OpenVINO IR.

        Args:
            export_type (ExportType): Export type should be ExportType.OPENVINO
            output_model (ModelEntity): The model entity in which to write the OpenVINO IR data

        Raises:
            Exception: If export_type is not ExportType.OPENVINO
        """
        assert export_type == ExportType.OPENVINO, f"Incorrect export_type={export_type}"
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = self.optimization_type

        # pylint: disable=no-member; need to refactor this
        logger.info("Exporting the OpenVINO model.")
        onnx_path = os.path.join(self.config.project.path, "onnx_model.onnx")
        self._export_to_onnx(onnx_path)
        optimize_command = "mo --input_model " + onnx_path + " --output_dir " + self.config.project.path
        subprocess.call(optimize_command, shell=True)
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

    def _model_info(self) -> Dict:
        """Return model info to save the model weights.

        Returns:
           Dict: Model info.
        """

        return {
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
        output_model.optimization_methods = self.optimization_methods

    def _set_metadata(self, output_model: ModelEntity):
        output_model.set_data("image_threshold", self.model.image_threshold.value.cpu().numpy().tobytes())
        output_model.set_data("pixel_threshold", self.model.pixel_threshold.value.cpu().numpy().tobytes())
        output_model.set_data("min", self.model.min_max.min.cpu().numpy().tobytes())
        output_model.set_data("max", self.model.min_max.max.cpu().numpy().tobytes())

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
        if os.path.exists(self.config.project.path):
            shutil.rmtree(self.config.project.path, ignore_errors=False)

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
