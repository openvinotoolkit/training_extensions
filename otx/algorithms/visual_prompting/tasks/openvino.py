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
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
from addict import Dict as ADDict
from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline
from omegaconf import OmegaConf

from otx.algorithms.visual_prompting.adapters.openvino import model_wrappers
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.config import (
    get_visual_promtping_config,
)
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.annotation import AnnotationSceneEntity
from otx.api.entities.dataset_item import DatasetItemEntity
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
from otx.api.usecases.exportable_code.inference import BaseInferencer
from otx.api.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from otx.api.utils.anomaly_utils import create_detection_annotation_from_anomaly_heatmap
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map

if find_spec("openvino") is not None:
    from openvino.inference_engine import IECore

from collections import defaultdict

import attr
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_zoo.model_api.models import Model

from otx.algorithms.common.utils.utils import get_default_async_reqs_num
from otx.algorithms.visual_prompting.adapters.openvino.adapter_wrappers import (
    VisualPromptingOpenvinoAdapter,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.dataset import (
    OTXVisualPromptingDataset,
    convert_polygon_to_mask,
    generate_bbox_from_mask,
)
from otx.algorithms.visual_prompting.configs.base import VisualPromptingBaseConfig
from otx.api.entities.image import Image
from otx.api.entities.label import LabelEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.shapes.polygon import Polygon
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    VisualPromptingToAnnotationConverter,
)

logger = get_logger()


class OpenVINOVisualPromptingInferencer(BaseInferencer):
    """Inferencer implementation for Visual Prompting using OpenVINO backend.
    
    Args:
        hparams (VisualPromptingBaseConfig): Hyper parameters that the model should use.
        label_schema (LabelSchemaEntity): LabelSchemaEntity that was used during model training.
        model_file (Union[str, bytes]): Path to model to load, `.xml`, `.bin` or `.onnx` file.
        weight_file (Union[str, bytes, None], optional): Path to weights to load, `.xml`, `.bin` or `.onnx` file. Defaults to None.
        device (str): Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        num_requests (int) : Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
    """
    def __init__(
        self,
        hparams: VisualPromptingBaseConfig,
        label_schema: LabelSchemaEntity,
        model_files: Dict[str, Union[str, Path]],
        weight_files: Optional[Dict[str, Union[str, Path, None]]] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):

        model_adapter = VisualPromptingOpenvinoAdapter(
            create_core(),
            model_files,
            weight_files,
            device=device,
            max_num_requests=num_requests,
            plugin_config={"PERFORMANCE_HINT": "THROUGHPUT"},
        )
        self.configuration = {
            **attr.asdict(
                hparams.postprocessing,
                filter=lambda attr, value: attr.name
                not in ["header", "description", "type", "visible_in_ui", "class_name"],
            )
        }
        self.model = Model.create_model(
            hparams.postprocessing.class_name.value,
            model_adapter,
            self.configuration,
            preload=True,
        )        
        self.converter = VisualPromptingToAnnotationConverter(label_schema)
        self.callback_exceptions: List[Exception] = []
        self.model.model_adapter.set_callback(self._async_callback)
        self.labels = label_schema.get_labels(include_empty=False)

    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Pre-process function of OpenVINO Visual Prompting Inferencer for image encoder."""
        return self.model.preprocess(image)

    def pre_process_prompt(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Pre-process function of OpenVINO Visual Prompting Inferencer for prompt encoder and mask decoder."""
        return self.model.preprocess_prompt(image)

    def post_process(
        self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Tuple[AnnotationSceneEntity, Any, Any]:
        """Post-process function of OpenVINO Visual Prompting Inferencer."""
        hard_prediction = self.model["decoder"].postprocess(prediction, metadata)
        soft_prediction = metadata["soft_prediction"]
        feature_vector = metadata["feature_vector"]
        predicted_scene = self.converter.convert_to_annotation(hard_prediction, metadata)

        return predicted_scene, feature_vector, soft_prediction

    def predict(self, dataset_item: DatasetItemEntity) -> Tuple[AnnotationSceneEntity, Any, Any]:
        """Perform a prediction for a given input image."""
        print("here!")
        pass
        # # forward image encoder
        # image, metadata = self.pre_process(dataset_item.numpy)
        # image_embeddings = self.forward_image_encoder(image)

        # # forward prompt encoder and mask decoder
        # width, height = dataset_item.width, dataset_item.height
        # inputs = {}
        # for annotation in dataset_item.get_annotations(labels=self.labels, include_empty=False, preserve_id=True):
        #     if isinstance(annotation.shape, Image):
        #         gt_mask = annotation.shape.numpy.astype(np.uint8)
        #     elif isinstance(annotation.shape, Polygon):
        #         gt_mask = convert_polygon_to_mask(annotation.shape, width, height)
        #     else:
        #         continue

        #     if gt_mask.sum() == 0:
        #         continue

        #     # generate bbox based on gt_mask
        #     bbox = generate_bbox_from_mask(gt_mask, width, height)

        #     # TODO (sungchul): generate random points from gt_mask

        #     # add labels
        #     labels.extend(annotation.get_labels(include_empty=False))
        
        #     predictions = self.forward_decoder(inputs)
        # return self.post_process(predictions, metadata)

    def forward(self, image: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Visual Prompting Inferencer."""
        return self.model["image_encoder"].infer_sync(image)

    def forward_decoder(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Visual Prompting Inferencer."""
        return self.model["decoder"].infer_sync(inputs)

    def enqueue_prediction(self, image: np.ndarray, id: int, result_handler: Any) -> None:
        """Runs async inference."""
        if not self.model.is_ready():
            self.model.await_any()
        image, metadata = self.pre_process(image)
        callback_data = id, metadata, result_handler
        self.model.infer_async(image, callback_data)

    def await_all(self) -> None:
        """Await all running infer requests if any."""
        self.model.await_all()

    def _async_callback(self, request: Any, callback_args: tuple) -> None:
        """Fetches the results of async inference."""
        try:
            res_copy_func, args = callback_args
            id, preprocessing_meta, result_handler = args
            prediction = res_copy_func(request)

            processed_prediciton = self.post_process(prediction, preprocessing_meta)
            result_handler(id, *processed_prediciton)

        except Exception as e:
            self.callback_exceptions.append(e)


class OpenVINOVisualPromptingTask(IInferenceTask, IEvaluationTask, IOptimizationTask, IDeploymentTask):
    """Task implementation for Visual Prompting using OpenVINO backend."""

    def __init__(self, task_environment: TaskEnvironment) -> None:
        self.task_environment = task_environment
        self.model = self.task_environment.model
        self.model_name = self.task_environment.model_template.model_template_id
        self.inferencer = self.load_inferencer()

        labels = task_environment.get_labels(include_empty=False)
        self._label_dictionary = dict(enumerate(labels, 1))
        template_file_path = self.task_environment.model_template.model_template_path
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))

    @property
    def hparams(self):
        """Hparams of OpenVINO Segmentation Task."""
        return self.task_environment.get_hyper_parameters(VisualPromptingBaseConfig)

    def load_inferencer(self) -> OpenVINOVisualPromptingInferencer:
        """Load OpenVINO Visual Prompting Inferencer."""
        if self.model is None:
            raise RuntimeError("load_inferencer failed, model is None")
        return OpenVINOVisualPromptingInferencer(
            self.hparams,
            self.task_environment.label_schema,
            {"image_encoder": self.model.get_data("sam_image_encoder.xml"), "decoder": self.model.get_data("sam_decoder.xml")},
            {"image_encoder": self.model.get_data("sam_image_encoder.bin"), "decoder": self.model.get_data("sam_decoder.bin")},
            num_requests=get_default_async_reqs_num(),
        )

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Infer function of OpenVINOSegmentationTask."""
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            enable_async_inference = inference_parameters.enable_async_inference
        else:
            update_progress_callback = default_progress_callback
            enable_async_inference = True

        def add_prediction(
            id: int,
            predicted_scene: AnnotationSceneEntity,
            feature_vector: Union[np.ndarray, None],
            soft_prediction: Union[np.ndarray, None],
        ):
            dataset_item = dataset[id]
            dataset_item.append_annotations(predicted_scene.annotations)

            if feature_vector is not None:
                feature_vector_media = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(feature_vector_media, model=self.model)

        total_time = 0.0
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            start_time = time.perf_counter()
            if enable_async_inference:
                self.inferencer.enqueue_prediction(dataset_item.numpy, i - 1, add_prediction)
            else:
                predicted_scene, feature_vector, soft_prediction = self.inferencer.predict(dataset_item.numpy)
                add_prediction(i - 1, predicted_scene, feature_vector, soft_prediction)
            end_time = time.perf_counter() - start_time
            total_time += end_time

            update_progress_callback(int(i / dataset_size * 100), None)

        self.inferencer.await_all()

        logger.info(f"Avg time per image: {total_time/len(dataset)} secs")
        logger.info(f"Total time: {total_time} secs")
        logger.info("Segmentation OpenVINO inference completed")

        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate function of OpenVINOSegmentationTask."""
        logger.info("Computing mDice")
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")

        output_resultset.performance = metrics.get_performance()

    def deploy(self, output_model: ModelEntity) -> None:
        """Deploy function of OpenVINOSegmentationTask."""
        logger.info("Deploying the model")
        if self.model is None:
            raise RuntimeError("deploy failed, model is None")

        work_dir = os.path.dirname(demo.__file__)
        parameters = {}
        parameters["type_of_model"] = self.hparams.postprocessing.class_name.value
        parameters["converter_type"] = "SEGMENTATION"
        parameters["model_parameters"] = self.inferencer.configuration
        parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(self.task_environment.label_schema)

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            arch.writestr(os.path.join("model", "model.xml"), self.model.get_data("openvino.xml"))
            arch.writestr(os.path.join("model", "model.bin"), self.model.get_data("openvino.bin"))
            arch.writestr(
                os.path.join("model", "config.json"),
                json.dumps(parameters, ensure_ascii=False, indent=4),
            )
            # model_wrappers files
            for root, _, files in os.walk(os.path.dirname(model_wrappers.__file__)):
                if "__pycache__" in root:
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    arch.write(
                        file_path,
                        os.path.join(
                            "python",
                            "model_wrappers",
                            file_path.split("model_wrappers/")[1],
                        ),
                    )
            # other python files
            arch.write(os.path.join(work_dir, "requirements.txt"), os.path.join("python", "requirements.txt"))
            arch.write(os.path.join(work_dir, "LICENSE"), os.path.join("python", "LICENSE"))
            arch.write(os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py"))
            arch.write(os.path.join(work_dir, "README.md"), os.path.join(".", "README.md"))
        output_model.exportable_code = zip_buffer.getvalue()
        logger.info("Deploying completed")

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        """Optimize function of OpenVINOSegmentationTask."""
        logger.info("Start POT optimization")
        if self.model is None:
            raise RuntimeError("POT optimize failed, model is None")

        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVino models")

        dataset = dataset.get_subset(Subset.TRAINING)
        data_loader = OTXOpenVinoDataLoader(dataset, self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            model_config = ADDict({"model_name": "openvino_model", "model": xml_path, "weights": bin_path})

            model = load_model(model_config)

            if get_nodes_by_type(model, ["FakeQuantize"]):
                raise RuntimeError("Model is already optimized by POT")

        if optimization_parameters is not None:
            optimization_parameters.update_progress(10, None)

        engine_config = ADDict({"device": "CPU"})

        optimization_config_path = os.path.join(self._base_dir, "pot_optimization_config.json")
        if os.path.exists(optimization_config_path):
            with open(optimization_config_path, encoding="UTF-8") as f_src:
                algorithms = ADDict(json.load(f_src))["algorithms"]
        else:
            algorithms = [ADDict({"name": "DefaultQuantization", "params": {"target_device": "ANY"}})]
        for algo in algorithms:
            algo.params.stat_subset_size = self.hparams.pot_parameters.stat_subset_size
            algo.params.shuffle_data = True
            if "Quantization" in algo["name"]:
                algo.params.preset = self.hparams.pot_parameters.preset.name.lower()

        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

        pipeline = create_pipeline(algorithms, engine)

        compressed_model = pipeline.run(model)

        compress_model_weights(compressed_model)

        if optimization_parameters is not None:
            optimization_parameters.update_progress(90, None)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            with open(os.path.join(tempdir, "model.xml"), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())

        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self.task_environment.label_schema),
        )

        # set model attributes for quantized model
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()

        if optimization_parameters is not None:
            optimization_parameters.update_progress(100, None)
        logger.info("POT optimization completed")

