"""OpenVINO Visual Prompting Task."""

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

import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union, Tuple
from zipfile import ZipFile

import numpy as np
from addict import Dict as ADDict
from anomalib.deploy import OpenVINOInferencer
from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline
from omegaconf import OmegaConf
from otx.api.entities.dataset_item import DatasetItemEntity

import otx.algorithms.anomaly.adapters.anomalib.exportable_code
from otx.algorithms.anomaly.adapters.anomalib.config import get_anomalib_config
from otx.algorithms.anomaly.adapters.anomalib.logger import get_logger
from otx.algorithms.anomaly.configs.base.configuration import BaseAnomalyConfig
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.model_template import TaskType
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)
from otx.api.usecases.exportable_code import demo
from otx.api.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from otx.api.utils.anomaly_utils import create_detection_annotation_from_anomaly_heatmap
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.config import (
    get_visual_promtping_config,
)
from pathlib import Path
from importlib.util import find_spec
if find_spec("openvino") is not None:
    from openvino.inference_engine import IECore
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets import OTXVisualPromptingDataset

logger = get_logger(__name__)


class OpenVINOTask(IInferenceTask, IEvaluationTask, IOptimizationTask, IDeploymentTask):
    """OpenVINO inference task.

    Args:
        task_environment (TaskEnvironment): task environment of the trained anomaly model
    """

    def __init__(self, task_environment: TaskEnvironment) -> None:
        logger.info("Initializing the OpenVINO task.")
        self.task_environment = task_environment
        self.task_type = self.task_environment.model_template.task_type

        template_file_path = task_environment.model_template.model_template_path
        self.base_dir = os.path.abspath(os.path.dirname(template_file_path))
        self.mode = "openvino"
        self.device = "CPU"

        self.config = self.get_config()

        # load models
        self.sam_image_encoder, self.sam_decoder = self.load_model(
            paths=dict(
                sam_image_encoder=(
                    self.task_environment.model.get_data("sam_image_encoder.xml"),
                    self.task_environment.model.get_data("sam_image_encoder.bin"),
                ),
                sam_decoder=(
                    self.task_environment.model.get_data("sam_decoder.xml"),
                    self.task_environment.model.get_data("sam_decoder.bin"),
                )
            ),
        )

        self.labels = self.task_environment.get_labels()

    def get_config(self) -> ADDict:
        """Get Visual Prompting Config from task environment.

        Returns:
            ADDict: Visual prompting config.
        """
        task_name = self.task_environment.model_template.name
        otx_config: ConfigurableParameters = self.task_environment.get_hyper_parameters()
        config = get_visual_promtping_config(
            task_name=task_name,
            otx_config=otx_config,
            config_dir=self.base_dir,
            mode=self.mode
        )
        return ADDict(OmegaConf.to_container(config))

    def load_model(self, paths: Dict[str, Tuple[bytes, bytes]]):
        """Load OpenVINO SAM models (image encoder, decoders).
        
        Args:
        
        """
        ie_core = IECore()
        sam_image_encoder_newtork = ie_core.read_network(
            model=paths["sam_image_encoder"][0],
            weights=paths["sam_image_encoder"][1],
            init_from_buffer=True)
        sam_image_encoder_executable_network = ie_core.load_network(
            network=sam_image_encoder_newtork, device_name=self.device)

        sam_decoder_network = ie_core.read_network(
            model=paths["sam_decoder"][0],
            weights=paths["sam_decoder"][1],
            init_from_buffer=True)
        sam_decoder_executable_network = ie_core.load_network(
            network=sam_decoder_network, device_name=self.device)

        return sam_image_encoder_executable_network, sam_decoder_executable_network

    def pre_process(self, dataset_item: DatasetItemEntity) -> np.ndarray:
        """Preprocess image for inference."""
        image = dataset_item.numpy


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
            update_progress_callback = inference_parameters.update_progress  # type: ignore

        # This always assumes that threshold is available in the task environment's model
        meta_data = self.get_metadata()
        self.transform = OTXVisualPromptingDataset
        for idx, dataset_item in enumerate(dataset):
            # preprocess inputs
            image, mask, bboxes, points = self.pre_process(dataset_item)

            # get image embeddings
            images = dataset_item["images"]
            image_embeddings = self.sam_image_encoder(images)

            # get result per prompt


            # # TODO: inferencer should return predicted label and mask
            # pred_label = image_result.pred_score >= 0.5
            # pred_mask = (image_result.anomaly_map >= 0.5).astype(np.uint8)
            # probability = image_result.pred_score if pred_label else 1 - image_result.pred_score
            # if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            #     label = self.anomalous_label if image_result.pred_score >= 0.5 else self.normal_label
            # elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            #     annotations = create_annotation_from_segmentation_map(
            #         pred_mask, image_result.anomaly_map.squeeze(), {0: self.normal_label, 1: self.anomalous_label}
            #     )
            #     dataset_item.append_annotations(annotations)
            #     label = self.normal_label if len(annotations) == 0 else self.anomalous_label
            # elif self.task_type == TaskType.ANOMALY_DETECTION:
            #     annotations = create_detection_annotation_from_anomaly_heatmap(
            #         pred_mask, image_result.anomaly_map.squeeze(), {0: self.normal_label, 1: self.anomalous_label}
            #     )
            #     dataset_item.append_annotations(annotations)
            #     label = self.normal_label if len(annotations) == 0 else self.anomalous_label
            # else:
            #     raise ValueError(f"Unknown task type: {self.task_type}")

            # dataset_item.append_labels([ScoredLabel(label=label, probability=float(probability))])
            # anomaly_map = (image_result.anomaly_map * 255).astype(np.uint8)
            # heatmap_media = ResultMediaEntity(
            #     name="Anomaly Map",
            #     type="anomaly_map",
            #     label=label,
            #     annotation_scene=dataset_item.annotation_scene,
            #     numpy=anomaly_map,
            # )
            # dataset_item.append_metadata_item(heatmap_media)
            # update_progress_callback(int((idx + 1) / len(dataset) * 100))

        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate the performance of the model.

        Args:
            output_resultset (ResultSetEntity): Result set storing ground truth and predicted dataset.
            evaluation_metric (Optional[str], optional): Evaluation metric. Defaults to None.
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

    def _get_optimization_algorithms_configs(self) -> List[ADDict]:
        raise NotImplementedError

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters],
    ):
        raise NotImplementedError

    def load_inferencer(self) -> OpenVINOInferencer:
        """Create the OpenVINO inferencer object.

        Returns:
            OpenVINOInferencer object
        """
        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot load weights.")

        return OpenVINOInferencer(
            path=(
                self.task_environment.model.get_data("openvino.xml"),
                self.task_environment.model.get_data("openvino.bin"),
            ),
            metadata=self.get_metadata(),
        )

    @staticmethod
    def __save_weights(path: str, data: bytes) -> None:
        """Write data to file.

        Args:
            path (str): Path of output file
            data (bytes): Data to write
        """
        with open(path, "wb") as file:
            file.write(data)

    @staticmethod
    def __load_weights(path: str, output_model: ModelEntity, key: str) -> None:
        """Load weights into output model.

        Args:
            path (str): Path to weights
            output_model (ModelEntity): Model to which the weights are assigned
            key (str): Key of the output model into which the weights are assigned
        """
        with open(path, "rb") as file:
            output_model.set_data(key, file.read())

    def _get_openvino_configuration(self) -> Dict[str, Any]:
        """Return configuration required by the exported model."""
        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot get configuration.")

        configuration = {
            "metadata": self.get_metadata(),
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
        raise NotImplementedError
