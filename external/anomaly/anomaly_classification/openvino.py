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
import os
import subprocess  # nosec
import sys
import tempfile
from shutil import copyfile, copytree
from typing import Any, Dict, List, Optional, Union
from zipfile import ZipFile

import numpy as np
from addict import Dict as ADDict
from anomalib.core.model.inference import OpenVINOInferencer
from anomalib.utils.post_process import anomaly_map_to_color_map
from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from ote_anomalib.config import get_anomalib_config
from ote_anomalib.exportable_code import AnomalyClassification
from ote_anomalib.logging import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.exportable_code import demo
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
    AnomalyClassificationToAnnotationConverter,
)
from ote_sdk.usecases.exportable_code.utils import set_proper_git_commit_hash
from ote_sdk.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)

logger = get_logger(__name__)


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


class OpenVINOAnomalyClassificationTask(IInferenceTask, IEvaluationTask, IOptimizationTask, IDeploymentTask):
    """
    OpenVINO inference task

    Args:
        task_environment (TaskEnvironment): task environment of the trained anomaly model
    """

    def __init__(self, task_environment: TaskEnvironment) -> None:
        logger.info("Initializing the OpenVINO task.")
        self.task_environment = task_environment
        self.config = self.get_config()
        self.inferencer = self.load_inferencer()
        self.annotation_converter = AnomalyClassificationToAnnotationConverter(self.task_environment.label_schema)
        template_file_path = task_environment.model_template.model_template_path
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))

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
        """Perform Inference.

        Args:
            dataset (DatasetEntity): Inference dataset
            inference_parameters (InferenceParameters): Inference parameters.

        Returns:
            DatasetEntity: Output dataset storing inference predictions.
        """
        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot access threshold to calculate labels.")

        logger.info("Start OpenVINO inference.")
        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress

        # This always assumes that threshold is available in the task environment's model
        meta_data = self.get_meta_data()
        for idx, dataset_item in enumerate(dataset):
            anomaly_map, pred_score = self.inferencer.predict(
                dataset_item.numpy, superimpose=False, meta_data=meta_data
            )
            annotations_scene = self.annotation_converter.convert_to_annotation(pred_score, meta_data)
            dataset_item.append_labels(annotations_scene.annotations[0].get_labels())
            anomaly_map = anomaly_map_to_color_map(anomaly_map, normalize=False)
            heatmap_media = ResultMediaEntity(
                name="Anomaly Map",
                type="anomaly_map",
                annotation_scene=dataset_item.annotation_scene,
                numpy=anomaly_map,
            )
            dataset_item.append_metadata_item(heatmap_media)
            update_progress_callback(int((idx + 1) / len(dataset) * 100))

        return dataset

    def get_meta_data(self):
        """Get Meta Data."""

        image_threshold = np.frombuffer(self.task_environment.model.get_data("image_threshold"), dtype=np.float32)
        min_value = np.frombuffer(self.task_environment.model.get_data("min"), dtype=np.float32)
        max_value = np.frombuffer(self.task_environment.model.get_data("max"), dtype=np.float32)
        meta_data = dict(
            image_threshold=image_threshold,
            pixel_threshold=image_threshold,  # re-use image threshold for pixel normalization
            min=min_value,
            max=max_value,
        )
        return meta_data

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate the performance of the model.

        Args:
            output_resultset (ResultSetEntity): Result set storing ground truth and predicted dataset.
            evaluation_metric (Optional[str], optional): Evaluation metric. Defaults to None.
        """
        metric = MetricsHelper.compute_f_measure(output_resultset)
        output_resultset.performance = metric.get_performance()

        # NOTE: This is for debugging purpose.
        for i, _ in enumerate(output_resultset.ground_truth_dataset):
            logger.info(
                "True vs Pred: %s %s - %3.2f",
                output_resultset.ground_truth_dataset[i].annotation_scene.annotations[0].get_labels()[0].name,
                output_resultset.prediction_dataset[i].annotation_scene.annotations[0].get_labels()[0].name,
                output_resultset.prediction_dataset[i].annotation_scene.annotations[0].get_labels()[0].probability,
            )
        logger.info("%s performance of the OpenVINO model: %3.2f", metric.f_measure.name, metric.f_measure.value)

    def _get_optimization_algorithms_configs(self) -> List[ADDict]:
        """Returns list of optimization algorithms configurations"""

        hparams = self.task_environment.get_hyper_parameters()

        optimization_config_path = os.path.join(self._base_dir, "pot_optimization_config.json")
        if os.path.exists(optimization_config_path):
            with open(optimization_config_path, encoding="UTF-8") as f_src:
                algorithms = ADDict(json.load(f_src))["algorithms"]
        else:
            algorithms = [
                ADDict({"name": "DefaultQuantization", "params": {"target_device": "ANY", "shuffle_data": True}})
            ]
        for algo in algorithms:
            algo.params.stat_subset_size = hparams.pot_parameters.stat_subset_size
            algo.params.shuffle_data = True
            if "Quantization" in algo["name"]:
                algo.params.preset = hparams.pot_parameters.preset.name.lower()

        return algorithms

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters],
    ):
        """Optimize the model.

        Args:
            optimization_type (OptimizationType): Type of optimization [POT or NNCF]
            dataset (DatasetEntity): Input Dataset.
            output_model (ModelEntity): Output model.
            optimization_parameters (Optional[OptimizationParameters]): Optimization parameters.

        Raises:
            ValueError: When the optimization type is not POT, which is the only support type at the moment.
        """
        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVINO models")

        logger.info("Starting POT optimization.")
        data_loader = OTEOpenVINOAnomalyDataloader(config=self.config, dataset=dataset, inferencer=self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")

            self.__save_weights(xml_path, self.task_environment.model.get_data("openvino.xml"))
            self.__save_weights(bin_path, self.task_environment.model.get_data("openvino.bin"))

            model_config = {
                "model_name": "openvino_model",
                "model": xml_path,
                "weights": bin_path,
            }
            model = load_model(model_config)

            if get_nodes_by_type(model, ["FakeQuantize"]):
                raise RuntimeError("Model is already optimized by POT")

        engine = IEEngine(config=ADDict({"device": "CPU"}), data_loader=data_loader, metric=None)
        pipeline = create_pipeline(algo_config=self._get_optimization_algorithms_configs(), engine=engine)
        compressed_model = pipeline.run(model)
        compress_model_weights(compressed_model)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            self.__load_weights(path=os.path.join(tempdir, "model.xml"), output_model=output_model, key="openvino.xml")
            self.__load_weights(path=os.path.join(tempdir, "model.bin"), output_model=output_model, key="openvino.bin")

        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))
        output_model.set_data("image_threshold", self.task_environment.model.get_data("image_threshold"))
        output_model.set_data("min", self.task_environment.model.get_data("min"))
        output_model.set_data("max", self.task_environment.model.get_data("max"))
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.task_environment.model = output_model
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> OpenVINOInferencer:
        """
        Create the OpenVINO inferencer object

        Returns:
            OpenVINOInferencer object
        """
        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot load weights.")
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

    def _get_openvino_configuration(self) -> Dict[str, Any]:
        """Return configuration required by the exported model."""
        # This always assumes that threshold is available in the task environment's model
        # cast is used to placate mypy
        configuration = {
            "image_threshold": np.frombuffer(
                self.task_environment.model.get_data("image_threshold"), dtype=np.float32
            ).item(),
            "min": np.frombuffer(self.task_environment.model.get_data("min"), dtype=np.float32).item(),
            "max": np.frombuffer(self.task_environment.model.get_data("max"), dtype=np.float32).item(),
            "labels": LabelSchemaMapper.forward(self.task_environment.label_schema),
            "threshold": 0.5,
        }
        if "transforms" not in self.config.keys():
            configuration["mean_values"] = list(np.array([0.485, 0.456, 0.406]) * 255)
            configuration["scale_values"] = list(np.array([0.229, 0.224, 0.225]) * 255)
        else:
            configuration["mean_values"] = self.config.transforms.mean
            configuration["scale_values"] = self.config.transforms.std
        return configuration

    def deploy(self, output_model: ModelEntity) -> None:
        """Exports the weights from ``output_model`` along with exportable code.

        Args:
            output_model (ModelEntity): Model with ``openvino.xml`` and ``.bin`` keys

        Raises:
            Exception: If ``task_environment.model`` is None
        """
        logger.info("Deploying Model")

        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot load weights.")

        work_dir = os.path.dirname(demo.__file__)
        parameters: Dict[str, Any] = {}
        parameters["type_of_model"] = "anomaly_classification"
        parameters["converter_type"] = "ANOMALY_CLASSIFICATION"
        parameters["model_parameters"] = self._get_openvino_configuration()
        name_of_package = "demo_package"

        with tempfile.TemporaryDirectory() as tempdir:
            copyfile(os.path.join(work_dir, "setup.py"), os.path.join(tempdir, "setup.py"))
            copyfile(os.path.join(work_dir, "requirements.txt"), os.path.join(tempdir, "requirements.txt"))
            set_proper_git_commit_hash(os.path.join(tempdir, "requirements.txt"))
            copytree(os.path.join(work_dir, name_of_package), os.path.join(tempdir, name_of_package))
            config_path = os.path.join(tempdir, name_of_package, "config.json")
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(parameters, file, ensure_ascii=False, indent=4)

            copyfile(inspect.getfile(AnomalyClassification), os.path.join(tempdir, name_of_package, "model.py"))

            # create wheel package
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(tempdir, "setup.py"),
                    "bdist_wheel",
                    "--dist-dir",
                    tempdir,
                    "clean",
                    "--all",
                ],
                check=True,
            )
            wheel_file_name = [f for f in os.listdir(tempdir) if f.endswith(".whl")][0]

            with ZipFile(os.path.join(tempdir, "openvino.zip"), "w") as arch:
                arch.writestr(os.path.join("model", "model.xml"), self.task_environment.model.get_data("openvino.xml"))
                arch.writestr(os.path.join("model", "model.bin"), self.task_environment.model.get_data("openvino.bin"))
                arch.write(os.path.join(tempdir, "requirements.txt"), os.path.join("python", "requirements.txt"))
                arch.write(os.path.join(work_dir, "README.md"), os.path.join("python", "README.md"))
                arch.write(os.path.join(work_dir, "LICENSE"), os.path.join("python", "LICENSE"))
                arch.write(os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py"))
                arch.write(os.path.join(tempdir, wheel_file_name), os.path.join("python", wheel_file_name))
            with open(os.path.join(tempdir, "openvino.zip"), "rb") as output_arch:
                output_model.exportable_code = output_arch.read()
        logger.info("Deployment completed.")
