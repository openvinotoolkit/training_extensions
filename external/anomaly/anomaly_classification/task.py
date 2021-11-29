"""
Anomaly Classification Task
"""

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
import logging
import os
import shutil
import struct
import subprocess
import tempfile
from glob import glob
from typing import Optional, Union

import torch
from anomalib.core.model import AnomalyModule
from anomalib.models import get_model
from omegaconf import DictConfig, ListConfig
from ote_anomalib.callbacks import InferenceCallback, ProgressCallback
from ote_anomalib.config import get_anomalib_config
from ote_anomalib.data import OTEAnomalyDataModule
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.metrics import Performance, ScoreMetric
from ote_sdk.entities.model import ModelEntity, ModelPrecision, ModelStatus
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from pytorch_lightning import Trainer

logger = logging.getLogger(__name__)


class AnomalyClassificationTask(ITrainingTask, IInferenceTask, IEvaluationTask, IExportTask, IUnload):
    """
    Base Anomaly Task for Training and Inference

    Args:
        task_environment (TaskEnvironment): OTE Task environment.
    """

    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.model_name = task_environment.model_template.name
        self.labels = task_environment.get_labels()

        # Hyperparameters.
        self.project_path: str = tempfile.mkdtemp(prefix="ote-anomalib")
        self.config = self.get_config()

        self.model = self.load_model(ote_model=task_environment.model)

        self.trainer: Trainer

    def get_config(self) -> Union[DictConfig, ListConfig]:
        """
        Get Anomalib Config from task environment

        Returns:
            Union[DictConfig, ListConfig]: Anomalib config
        """
        hyper_parameters = self.task_environment.get_hyper_parameters()
        config = get_anomalib_config(task_name=self.model_name, ote_config=hyper_parameters)
        config.dataset.task = "classification"
        config.project.path = self.project_path
        return config

    def load_model(self, ote_model: Optional[ModelEntity]) -> AnomalyModule:
        """
        Create and Load Anomalib Module from OTE Model.
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

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
    ):
        """
        Train the anomaly task
        """
        config = self.get_config()
        datamodule = OTEAnomalyDataModule(config=config, dataset=dataset)
        callbacks = [ProgressCallback(parameters=train_parameters)]

        self.trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks)
        self.trainer.fit(model=self.model, datamodule=datamodule)

        self.save_model(output_model)

    def save_model(self, output_model: ModelEntity):
        """
        Save the model after training is completed.
        """
        config = self.get_config()
        model_info = {
            "model": self.model.state_dict(),
            "config": config,
            "label_schema": self.task_environment.label_schema,
            "VERSION": 1,
        }
        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        # store computed threshold
        output_model.set_data("threshold", bytes(struct.pack("f", self.model.threshold.item())))

        f1_score = self.model.results.performance["image_f1_score"]
        output_model.performance = Performance(score=ScoreMetric(name="F1 Score", value=f1_score))
        output_model.precision = [ModelPrecision.FP32]
        output_model.model_status = ModelStatus.SUCCESS

    def cancel_training(self):
        """
        Cancel the training `after_batch_end`. This terminates the training; however validation is
        still performed.
        """
        logger.info("Cancel training requested.")
        self.trainer.should_stop = True

        # The runner periodically checks `.stop_training` file to ensure if cancellation is requested.
        cancel_training_file_path = os.path.join(self.config.project.path, ".stop_training")
        with open(file=cancel_training_file_path, mode="a", encoding="utf-8"):
            pass

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        """
        Perform inference on a dataset.
        """
        config = self.get_config()
        datamodule = OTEAnomalyDataModule(config=config, dataset=dataset)

        # Callbacks.
        progress = ProgressCallback(parameters=inference_parameters)
        inference = InferenceCallback(dataset, self.labels)
        callbacks = [progress, inference]

        self.trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks)
        self.trainer.predict(model=self.model, datamodule=datamodule)
        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """
        Evaluate the performance on a result set.
        """
        f_measure_metrics = MetricsHelper.compute_f_measure(output_resultset)
        output_resultset.performance = f_measure_metrics.get_performance()
        logger.info("F-measure after evaluation: %d", f_measure_metrics.f_measure.value)

    def export(self, export_type: ExportType, output_model: ModelEntity):
        """Export model to OpenVINO IR

        Args:
            export_type (ExportType): Export type should be ExportType.OPENVINO
            output_model (ModelEntity): The model entity in which to write the OpenVINO IR data

        Raises:
            Exception: If export_type is not ExportType.OPENVINO
        """
        assert export_type == ExportType.OPENVINO

        # pylint: disable=no-member; need to refactor this
        height, width = self.config.model.input_size
        onnx_path = os.path.join(self.config.project.path, "onnx_model.onnx")
        torch.onnx.export(
            model=self.model.model,
            args=torch.zeros((1, 3, height, width)).to(self.model.device),
            f=onnx_path,
            opset_version=11,
        )
        optimize_command = "mo --input_model " + onnx_path + " --output_dir " + self.config.project.path
        subprocess.call(optimize_command, shell=True)
        bin_file = glob(os.path.join(self.config.project.path, "*.bin"))[0]
        xml_file = glob(os.path.join(self.config.project.path, "*.xml"))[0]
        with open(bin_file, "rb") as file:
            output_model.set_data("openvino.bin", file.read())
        with open(xml_file, "rb") as file:
            output_model.set_data("openvino.xml", file.read())

    @staticmethod
    def _is_docker() -> bool:
        """
        Checks whether the task runs in docker container

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
        """
        Unload the task
        """
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
