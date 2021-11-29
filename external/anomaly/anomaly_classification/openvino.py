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

import inspect
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
from shutil import copyfile, copytree
from typing import Optional, Union

from addict import Dict as ADDict

from anomalib.core.model import model_wrappers
from anomalib.core.model.inference import OpenVINOInferencer

from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline

from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig

import ote_sdk.usecases.exportable_code.demo as demo
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters, default_progress_callback
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
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

    def __init__(self, config: Union[DictConfig, ListConfig], dataset: DatasetEntity, inferencer: OpenVINOInferencer):
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
        config (Union[DictConfig, ListConfig]): configuration file
    """

    def __init__(
        self,
        task_environment: TaskEnvironment,
        config: Union[DictConfig, ListConfig],
    ) -> None:
        self.task_environment = task_environment
        self.config = config
        self.model = self.task_environment.model
        self.model_name = task_environment.model_template.name
        labels = task_environment.get_labels()
        self.normal_label = [label for label in labels if label.name == "normal"][0]
        self.anomalous_label = [label for label in labels if label.name == "anomalous"][0]
        self.inferencer = self.load_inferencer()

    @property
    def hparams(self):
        return self.task_environment.get_hyper_parameters()

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        logger.info('Start OpenVINO inference')
        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            predicted_scene = self.inferencer.predict(dataset_item.numpy, superimpose=False)
            dataset_item.append_annotations(predicted_scene.annotations)
            update_progress_callback(int(i / dataset_size * 100))

        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        output_resultset.performance = MetricsHelper.compute_f_measure(output_resultset).get_performance()

    def deploy(self,
               output_path: str):
        work_dir = os.path.dirname(demo.__file__)
        model_file = inspect.getfile(type(self.inferencer.model))
        parameters = {}
        parameters['name_of_model'] = self.model_name
        parameters['type_of_model'] = self.hparams.inference_parameters.class_name.value
        parameters['converter_type'] = 'ANOMALY_CLASSIFICATION'
        parameters['model_parameters'] = self.inferencer.configuration
        name_of_package = parameters['name_of_model'].lower()
        with tempfile.TemporaryDirectory() as tempdir:
            copyfile(os.path.join(work_dir, "setup.py"), os.path.join(tempdir, "setup.py"))
            copyfile(os.path.join(work_dir, "requirements.txt"), os.path.join(tempdir, "requirements.txt"))
            copytree(os.path.join(work_dir, "demo_package"), os.path.join(tempdir, name_of_package))
            xml_path = os.path.join(tempdir, name_of_package, "model.xml")
            bin_path = os.path.join(tempdir, name_of_package, "model.bin")
            config_path = os.path.join(tempdir, name_of_package, "config.json")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))
            with open(config_path, "w") as f:
                json.dump(parameters, f)
            # generate model.py
            if (inspect.getmodule(self.inferencer.model) in
               [module[1] for module in inspect.getmembers(model_wrappers, inspect.ismodule)]):
                copyfile(model_file, os.path.join(tempdir, name_of_package, "model.py"))
            # create wheel package
            subprocess.run([sys.executable, os.path.join(tempdir, "setup.py"), 'bdist_wheel',
                            '--dist-dir', output_path, 'clean', '--all'])

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

        algorithms = [
            {
                "name": "DefaultQuantization",
                "params": {
                    "target_device": "ANY",
                    "preset": self.hparams.pot_parameters.preset.name.lower(),
                    "stat_subset_size": min(self.hparams.pot_parameters.stat_subset_size, len(data_loader)),
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
        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot access threshold to calculate labels.")
        # This always assumes that threshold is available in the task environment's model
        threshold = struct.unpack("f", (self.task_environment.model.get_data("threshold")))
        return OpenVINOInferencer(
            config=self.config,
            hparams=self.task_environment.get_hyper_parameters(),
            threshold=threshold[0],
            labels=[self.normal_label, self.anomalous_label],
            path=(
                self.model.get_data("openvino.xml"),
                self.model.get_data("openvino.bin"),
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
