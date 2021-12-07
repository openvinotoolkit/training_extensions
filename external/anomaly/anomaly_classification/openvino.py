"""
OpenVINO Anomaly Task
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

import logging
import os
import struct
import tempfile
from typing import Optional, Union

from addict import Dict as ADDict
from anomalib.core.model.inference import OpenVINOInferencer
from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from ote_anomalib.config import get_anomalib_config
from ote_anomalib.data import LabelNames
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)

logger = logging.getLogger(__name__)


class OTEOpenVINOAnomalyDataloader(DataLoader):
    """
    Dataloader for loading OTE dataset into OTE OpenVINO Inferencer

    Args:
        dataset (DatasetEntity): OTE dataset entity
        inferencer (OpenVINOInferencer): OpenVINO Inferencer
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        dataset: DatasetEntity,
        inferencer: OpenVINOInferencer,
    ):
        super().__init__(config=config)
        self.dataset = dataset
        self.inferencer = inferencer

    def __getitem__(self, index):
        image = self.dataset[index].numpy
        annotation = self.dataset[index].annotation_scene
        inputs = self.inferencer.pre_process(image)

        return (index, annotation), inputs

    def __len__(self):
        return len(self.dataset)


class OpenVINOAnomalyClassificationTask(IInferenceTask, IEvaluationTask, IOptimizationTask):
    """
    OpenVINO inference task

    Args:
        task_environment (TaskEnvironment): task environment of the trained anomaly model
    """

    def __init__(self, task_environment: TaskEnvironment) -> None:
        self.task_environment = task_environment
        self.config = self.get_config()
        self.inferencer = self.load_inferencer()
        labels = task_environment.get_labels()
        self.normal_label = [label for label in labels if label.name == LabelNames.normal][0]
        self.anomalous_label = [label for label in labels if label.name == LabelNames.anomalous][0]

    def get_config(self) -> Union[DictConfig, ListConfig]:
        """
        Get Anomalib Config from task environment

        Returns:
            Union[DictConfig, ListConfig]: Anomalib config
        """
        task_name = self.task_environment.model_template.name
        ote_config = self.task_environment.get_hyper_parameters()
        config = get_anomalib_config(task_name=task_name, ote_config=ote_config)
        return config

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot access threshold to calculate labels.")
        # This always assumes that threshold is available in the task environment's model
        threshold = struct.unpack("f", (self.task_environment.model.get_data("threshold")))
        for dataset_item in dataset:
            anomaly_map = self.inferencer.predict(dataset_item.numpy, superimpose=False)
            pred_score = anomaly_map.reshape(-1).max()
            pred_label = pred_score >= threshold
            assigned_label = self.anomalous_label if pred_label else self.normal_label
            shape = Annotation(Rectangle(x1=0, y1=0, x2=1, y2=1), labels=[ScoredLabel(assigned_label, probability=float(pred_score))])
            dataset_item.append_annotations([shape])

        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        output_resultset.performance = MetricsHelper.compute_f_measure(output_resultset).get_performance()

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters],
    ):
        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVINO models")

        data_loader = OTEOpenVINOAnomalyDataloader(config=self.config, dataset=dataset, inferencer=self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")

            self.__save_weights(xml_path, self.task_environment.model.get_data("openvino.xml"))
            self.__save_weights(bin_path, self.task_environment.model.get_data("openvino.bin"))

            model_config = {"model_name": "openvino_model", "model": xml_path, "weights": bin_path}
            model = load_model(model_config)

            if get_nodes_by_type(model, ["FakeQuantize"]):
                logger.warning("Model is already optimized by POT")
                output_model.model_status = ModelStatus.FAILED
                return

        hparams = self.task_environment.get_hyper_parameters()

        algorithms = [
            {
                "name": "DefaultQuantization",
                "params": {
                    "target_device": "ANY",
                    "preset": hparams.pot_parameters.preset.name.lower(),
                    "stat_subset_size": min(hparams.pot_parameters.stat_subset_size, len(data_loader)),
                },
            }
        ]

        engine = IEEngine(config=ADDict({"device": "CPU"}), data_loader=data_loader, metric=None)

        compressed_model = create_pipeline(algorithms, engine).run(model)
        compress_model_weights(compressed_model)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            self.__load_weights(path=os.path.join(tempdir, "model.xml"), output_model=output_model, key="openvino.xml")
            self.__load_weights(path=os.path.join(tempdir, "model.bin"), output_model=output_model, key="openvino.bin")
        output_model.model_status = ModelStatus.SUCCESS

        self.task_environment.model = output_model
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> OpenVINOInferencer:
        """
        Create the OpenVINO inferencer object

        Returns:
            OpenVINOInferencer object
        """
        return OpenVINOInferencer(
            config=self.config,
            path=(
                self.task_environment.model.get_data("openvino.xml"),
                self.task_environment.model.get_data("openvino.bin"),
            ),
        )

    @staticmethod
    def __save_weights(path: str, data: bytes) -> None:
        """Write data to file

        Args:
            path (str): Path of output file
            data (bytes): Data to write
        """
        with open(path, "wb") as file:
            file.write(data)

    @staticmethod
    def __load_weights(path: str, output_model: ModelEntity, key: str) -> None:
        """
        Load weights into output model

        Args:
            path (str): Path to weights
            output_model (ModelEntity): Model to which the weights are assigned
            key (str): Key of the output model into which the weights are assigned
        """
        with open(path, "rb") as file:
            output_model.set_data(key, file.read())
