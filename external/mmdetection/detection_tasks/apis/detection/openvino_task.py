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

import attr
import copy
import io
import json
import multiprocessing
import numpy as np
import os
import ote_sdk.usecases.exportable_code.demo as demo
import tempfile
import warnings
from addict import Dict as ADDict
from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_zoo.model_api.models import Model, utils
from ote_sdk.entities.annotation import AnnotationSceneEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from ote_sdk.configuration.helper.utils import config_to_bytes
from ote_sdk.entities.model_template import TaskType
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.exportable_code.inference import BaseInferencer
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
    DetectionToAnnotationConverter,
    IPredictionToAnnotationConverter,
    MaskToAnnotationConverter,
    RotatedRectToAnnotationConverter,
)
from ote_sdk.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from ote_sdk.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from ote_sdk.utils import Tiler
from ote_sdk.utils.vis_utils import get_actmap
from typing import Any, Dict, Optional, Tuple, Union, List
from zipfile import ZipFile

from mmdet.utils.logger import get_root_logger
from .configuration import OTEDetectionConfig
from . import model_wrappers

from mmcv.ops import nms

logger = get_root_logger()


def multiclass_nms(
    scores: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    iou_threshold=0.45,
    max_num=200,
):
    """ Multi-class NMS

    strategy: in order to perform NMS independently per class,
    we add an offset to all the boxes. The offset is dependent
    only on the class idx, and is large enough so that boxes
    from different classes do not overlap

    Args:
        scores (np.ndarray): box scores
        labels (np.ndarray): box label indices
        boxes (np.ndarray): box coordinates
        iou_threshold (float, optional): IoU threshold. Defaults to 0.45.
        max_num (int, optional): Max number of objects filter. Defaults to 200.

    Returns:
        _type_: _description_
    """
    max_coordinate = boxes.max()
    offsets = labels.astype(boxes.dtype) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    dets, keep = nms(boxes_for_nms, scores, iou_threshold)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    return dets, keep


class BaseInferencerWithConverter(BaseInferencer):
    @check_input_parameters_type()
    def __init__(
        self,
        configuration: dict,
        model: Model,
        converter: IPredictionToAnnotationConverter,
    ) -> None:
        self.configuration = configuration
        self.model = model
        self.converter = converter

    @check_input_parameters_type()
    def pre_process(
        self, image: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return self.model.preprocess(image)

    @check_input_parameters_type()
    def post_process(
        self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        detections = self.model.postprocess(prediction, metadata)
        return self.converter.convert_to_annotation(detections, metadata)

    @check_input_parameters_type()
    def predict(
        self, image: np.ndarray
    ) -> Tuple[AnnotationSceneEntity, np.ndarray, np.ndarray]:
        image, metadata = self.pre_process(image)
        raw_predictions = self.forward(image)
        predictions = self.post_process(raw_predictions, metadata)
        if (
            "feature_vector" not in raw_predictions
            or "saliency_map" not in raw_predictions
        ):
            warnings.warn(
                "Could not find Feature Vector and Saliency Map in OpenVINO output. "
                "Please rerun OpenVINO export or retrain the model."
            )
            features = [None, None]
        else:
            features = [
                raw_predictions["feature_vector"].reshape(-1),
                raw_predictions["saliency_map"],
            ]
        return predictions, features

    @check_input_parameters_type()
    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.model.infer_sync(inputs)


class OpenVINODetectionInferencer(BaseInferencerWithConverter):
    @check_input_parameters_type()
    def __init__(
        self,
        hparams: OTEDetectionConfig,
        label_schema: LabelSchemaEntity,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        """
        Inferencer implementation for OTEDetection using OpenVINO backend.

        :param hparams: Hyper parameters that the model should use.
        :param label_schema: LabelSchemaEntity that was used during model training.
        :param model_file: Path OpenVINO IR model definition file.
        :param weight_file: Path OpenVINO IR model weights file.
        :param device: Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        :param num_requests: Maximum number of requests that the inferencer can make. Defaults to 1.

        """

        model_adapter = OpenvinoAdapter(
            create_core(),
            model_file,
            weight_file,
            device=device,
            max_num_requests=num_requests,
        )
        configuration = {
            **attr.asdict(
                hparams.postprocessing,
                filter=lambda attr, value: attr.name
                not in ["header", "description", "type", "visible_in_ui"],
            )
        }
        model = Model.create_model(
            "OTE_SSD", model_adapter, configuration, preload=True
        )
        converter = DetectionToAnnotationConverter(label_schema)

        super().__init__(configuration, model, converter)

    def convert2array(self, detections: List[utils.Detection]) -> np.ndarray:
        """ Convert list of OpenVINO Detection to a numpy array

        Args:
            detections (List[utils.Detection]): List of OpenVINO Detection containing score, id, xmin, ymin, xmax, ymax

        Returns:
            np.ndarray: numpy array with [label, confidence, x1, y1, x2, y2]
        """
        scores = np.empty((0, 1), dtype=np.float32)
        labels = np.empty((0, 1), dtype=np.uint32)
        boxes = np.empty((0, 4), dtype=np.float32)
        for det in detections:
            if (det.xmax - det.xmin) * (det.ymax - det.ymin) < 1.0:
                continue
            scores = np.append(scores, [[det.score]], axis=0)
            labels = np.append(labels, [[det.id]], axis=0)
            boxes = np.append(
                boxes,
                [[float(det.xmin), float(det.ymin), float(det.xmax), float(det.ymax)]],
                axis=0,
            )
        detections = np.concatenate((labels, scores, boxes), -1)
        return detections

    @check_input_parameters_type()
    def post_process(
        self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        detections = self.model.postprocess(prediction, metadata)
        detections = self.convert2array(detections)
        detections[:, 2:] /= np.tile(metadata["original_shape"][1::-1], 2)
        return self.converter.convert_to_annotation(detections)

    @check_input_parameters_type()
    def predict_tile(
        self, image: np.ndarray, tile_size: int, overlap: float, max_number: int
    ) -> Tuple[AnnotationSceneEntity, np.ndarray, np.ndarray]:
        """ Run prediction by tiling image to small patches

        Args:
            image (np.ndarray): input image
            tile_size (int): tile crop size
            overlap (float): overlap ratio between tiles
            max_number (int): max number of predicted objects allowed

        Returns:
            detections: AnnotationSceneEntity
            features: list including saliency map and feature vector
        """
        detections = np.empty((0, 6), dtype=np.float32)

        tiler = Tiler(tile_size=tile_size, overlap=overlap)

        original_shape = image.shape
        features = [None, None]
        for i, coord in enumerate(tiler.tile(image)):
            x1, y1, x2, y2 = coord
            tile_dict, tile_meta = self.pre_process(image[y1:y2, x1:x2])
            raw_predictions = self.forward(tile_dict)
            output = self.model.postprocess(raw_predictions, tile_meta)
            # cache full image feature vector and saliency map at 0 index
            if i == 0:
                if (
                    "feature_vector" in raw_predictions
                    or "saliency_map" in raw_predictions
                ):
                    features = [
                        raw_predictions["feature_vector"].reshape(-1),
                        raw_predictions["saliency_map"],
                    ]
            output = self.convert2array(output)
            output[:, 2:] += np.tile([x1, y1], 2)
            detections = np.append(detections, output, axis=0)

        if np.prod(detections):
            scores = detections[:, 0]
            labels = detections[:, 1]
            boxes = detections[:, 2:]
            _, keep = multiclass_nms(scores, labels, boxes, max_num=max_number)
            detections = detections[keep]

        # normalize coordinates between 0 and 1
        detections[:, 2:] /= np.tile(original_shape[1::-1], 2)
        detections = self.converter.convert_to_annotation(detections)
        return detections, features


class OpenVINOMaskInferencer(BaseInferencerWithConverter):
    @check_input_parameters_type()
    def __init__(
        self,
        hparams: OTEDetectionConfig,
        label_schema: LabelSchemaEntity,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        model_adapter = OpenvinoAdapter(
            create_core(),
            model_file,
            weight_file,
            device=device,
            max_num_requests=num_requests,
        )

        configuration = {
            **attr.asdict(
                hparams.postprocessing,
                filter=lambda attr, value: attr.name
                not in ["header", "description", "type", "visible_in_ui"],
            )
        }

        model = Model.create_model(
            "ote_maskrcnn", model_adapter, configuration, preload=True
        )

        converter = MaskToAnnotationConverter(label_schema)

        super().__init__(configuration, model, converter)

    @check_input_parameters_type()
    def predict_tile(
        self, image: np.ndarray, tile_size: int, overlap: float, max_number: int
    ) -> Tuple[AnnotationSceneEntity, np.ndarray, np.ndarray]:
        """ Run prediction by tiling image to small patches

        Args:
            image (np.ndarray): input image
            tile_size (int): tile crop size
            overlap (float): overlap ratio between tiles
            max_number (int): max number of predicted objects allowed

        Returns:
            detections: AnnotationSceneEntity
            features: list including saliency map and feature vector
        """
        scores = np.empty((0), dtype=np.float32)
        labels = np.empty((0), dtype=np.uint32)
        boxes = np.empty((0, 4), dtype=np.float32)
        masks = []

        tiler = Tiler(tile_size=tile_size, overlap=overlap)

        original_shape = image.shape
        tile_meta = {}
        features = [None, None]
        for i, coord in enumerate(tiler.tile(image)):
            x1, y1, x2, y2 = coord
            tile_dict, tile_meta = self.model.preprocess(image[y1:y2, x1:x2])
            tile_meta["resize_mask"] = False
            raw_predictions = self.model.infer_sync(tile_dict)
            # cache full image feature vector and saliency map at 0 index
            if i == 0:
                if (
                    "feature_vector" in raw_predictions
                    or "saliency_map" in raw_predictions
                ):
                    features = [
                        raw_predictions["feature_vector"].reshape(-1),
                        raw_predictions["saliency_map"],
                    ]
            tile_scores, tile_labels, tile_boxes, tile_masks = self.model.postprocess(raw_predictions, tile_meta)
            tile_boxes += np.tile([x1, y1], 2)
            scores = np.append(scores, tile_scores, axis=0)
            labels = np.append(labels, tile_labels, axis=0)
            boxes = np.append(boxes, tile_boxes, axis=0)
            masks.extend(tile_masks)

        assert len(scores) == len(labels) == len(boxes) == len(masks)
        if np.prod(scores):
            _, keep = multiclass_nms(scores, labels, boxes, max_num=max_number)
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            masks = [masks[keep_idx] for keep_idx in keep]

            for i in range(len(boxes)):
                masks[i] = self.model._segm_postprocess(
                    boxes[i], masks[i], *original_shape[:-1]
                )

        tile_meta["original_shape"] = original_shape
        detections = scores, labels, boxes, masks
        detections = self.converter.convert_to_annotation(detections, tile_meta)
        return detections, features


class OpenVINORotatedRectInferencer(BaseInferencerWithConverter):
    @check_input_parameters_type()
    def __init__(
        self,
        hparams: OTEDetectionConfig,
        label_schema: LabelSchemaEntity,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        model_adapter = OpenvinoAdapter(
            create_core(),
            model_file,
            weight_file,
            device=device,
            max_num_requests=num_requests,
        )

        configuration = {
            **attr.asdict(
                hparams.postprocessing,
                filter=lambda attr, value: attr.name
                not in ["header", "description", "type", "visible_in_ui"],
            )
        }

        model = Model.create_model(
            "ote_maskrcnn", model_adapter, configuration, preload=True
        )

        converter = RotatedRectToAnnotationConverter(label_schema)

        super().__init__(configuration, model, converter)


class OTEOpenVinoDataLoader(DataLoader):
    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def __init__(self, dataset: DatasetEntity, inferencer: BaseInferencer):
        self.dataset = dataset
        self.inferencer = inferencer

    @check_input_parameters_type()
    def __getitem__(self, index: int):
        image = self.dataset[index].numpy
        annotation = self.dataset[index].annotation_scene
        inputs, metadata = self.inferencer.pre_process(image)

        return (index, annotation), inputs, metadata

    def __len__(self):
        return len(self.dataset)


class OpenVINODetectionTask(
    IDeploymentTask, IInferenceTask, IEvaluationTask, IOptimizationTask
):
    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        logger.info("Loading OpenVINO OTEDetectionTask")
        self.task_environment = task_environment
        self.model = self.task_environment.model
        self.task_type = self.task_environment.model_template.task_type
        self.confidence_threshold: float = 0.0
        self.config = self.load_config()
        self.inferencer = self.load_inferencer()
        logger.info("OpenVINO task initialization completed")

    @property
    def hparams(self):
        return self.task_environment.get_hyper_parameters(OTEDetectionConfig)

    def load_config(self) -> Dict:
        """ Load configurable parameters from model adapter

        Returns:
            Dict: config dictionary
        """
        if self.model.get_data("config.json"):
            return json.loads(self.model.get_data("config.json"))
        return dict()

    def load_inferencer(
        self,
    ) -> Union[OpenVINODetectionInferencer, OpenVINOMaskInferencer]:
        _hparams = copy.deepcopy(self.hparams)
        self.confidence_threshold = float(
            np.frombuffer(
                self.model.get_data("confidence_threshold"), dtype=np.float32
            )[0]
        )
        _hparams.postprocessing.confidence_threshold = self.confidence_threshold
        _hparams.tiling_parameters.enable_tiling = self.config["tiling_parameters"][
            "enable_tiling"
        ]["value"]
        args = [
            _hparams,
            self.task_environment.label_schema,
            self.model.get_data("openvino.xml"),
            self.model.get_data("openvino.bin"),
        ]
        if self.task_type == TaskType.DETECTION:
            return OpenVINODetectionInferencer(*args)
        if self.task_type == TaskType.INSTANCE_SEGMENTATION:
            return OpenVINOMaskInferencer(*args)
        if self.task_type == TaskType.ROTATED_DETECTION:
            return OpenVINORotatedRectInferencer(*args)
        raise RuntimeError(f"Unknown OpenVINO Inferencer TaskType: {self.task_type}")

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        logger.info("Start OpenVINO inference")
        update_progress_callback = default_progress_callback
        add_saliency_map = True
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            add_saliency_map = not inference_parameters.is_evaluation

        if self.config and self.config["tiling_parameters"]["enable_tiling"]["value"]:
            tile_size = self.config["tiling_parameters"]["tile_size"]["value"]
            tile_overlap = self.config["tiling_parameters"]["tile_overlap"]["value"]
            max_number = self.config["tiling_parameters"]["tile_max_number"]["value"]
            logger.info("Run inference with tiling")

        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            if self.config["tiling_parameters"]["enable_tiling"]["value"]:
                predicted_scene, features = self.inferencer.predict_tile(
                    dataset_item.numpy,
                    tile_size=tile_size,
                    overlap=tile_overlap,
                    max_number=max_number,
                )
            else:
                predicted_scene, features = self.inferencer.predict(dataset_item.numpy)

            dataset_item.append_annotations(predicted_scene.annotations)
            update_progress_callback(int(i / dataset_size * 100))
            feature_vector, saliency_map = features
            if feature_vector is not None:
                representation_vector = TensorEntity(
                    name="representation_vector", numpy=feature_vector.reshape(-1)
                )
                dataset_item.append_metadata_item(
                    representation_vector, model=self.model
                )

            if add_saliency_map and saliency_map is not None:
                saliency_map = get_actmap(
                    saliency_map, (dataset_item.width, dataset_item.height)
                )
                saliency_map_media = ResultMediaEntity(
                    name="Saliency Map",
                    type="saliency_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=saliency_map,
                    roi=dataset_item.roi,
                )
                dataset_item.append_metadata_item(saliency_map_media, model=self.model)
        logger.info("OpenVINO inference completed")
        return dataset

    @check_input_parameters_type()
    def evaluate(
        self,
        output_result_set: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        logger.info("Start OpenVINO metric evaluation")
        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, but parameter is ignored. Use F-measure instead."
            )
        output_result_set.performance = MetricsHelper.compute_f_measure(
            output_result_set
        ).get_performance()
        logger.info("OpenVINO metric evaluation completed")

    @check_input_parameters_type()
    def deploy(self, output_model: ModelEntity) -> None:
        logger.info("Deploying the model")

        work_dir = os.path.dirname(demo.__file__)
        parameters = {}
        parameters["type_of_model"] = self.inferencer.model.__model__
        parameters["converter_type"] = str(self.task_type)
        parameters["model_parameters"] = self.inferencer.configuration
        parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(
            self.task_environment.label_schema
        )
        parameters["tiling_parameters"] = self.config["tiling_parameters"]

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            arch.writestr(
                os.path.join("model", "model.xml"), self.model.get_data("openvino.xml")
            )
            arch.writestr(
                os.path.join("model", "model.bin"), self.model.get_data("openvino.bin")
            )
            arch.writestr(
                os.path.join("model", "config.json"),
                json.dumps(parameters, ensure_ascii=False, indent=4),
            )
            # model_wrappers files
            for root, dirs, files in os.walk(os.path.dirname(model_wrappers.__file__)):
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
            # python files
            arch.write(
                os.path.join(work_dir, "requirements.txt"),
                os.path.join("python", "requirements.txt"),
            )
            arch.write(
                os.path.join(work_dir, "LICENSE"), os.path.join("python", "LICENSE")
            )
            arch.write(
                os.path.join(work_dir, "README.md"), os.path.join("python", "README.md")
            )
            arch.write(
                os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py")
            )
        output_model.exportable_code = zip_buffer.getvalue()
        logger.info("Deploying completed")

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        logger.info("Start POT optimization")

        if optimization_type is not OptimizationType.POT:
            raise ValueError(
                "POT is the only supported optimization type for OpenVino models"
            )

        data_loader = OTEOpenVinoDataLoader(dataset, self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            model_config = ADDict(
                {"model_name": "openvino_model", "model": xml_path, "weights": bin_path}
            )

            model = load_model(model_config)

            if get_nodes_by_type(model, ["FakeQuantize"]):
                raise RuntimeError("Model is already optimized by POT")

        if optimization_parameters is not None:
            optimization_parameters.update_progress(10)

        engine_config = ADDict(
            {
                "device": "CPU",
                "stat_requests_number": min(
                    self.hparams.pot_parameters.stat_requests_number,
                    multiprocessing.cpu_count(),
                ),
            }
        )

        stat_subset_size = self.hparams.pot_parameters.stat_subset_size
        preset = self.hparams.pot_parameters.preset.name.lower()

        algorithms = [
            {
                "name": "DefaultQuantization",
                "params": {
                    "target_device": "ANY",
                    "preset": preset,
                    "stat_subset_size": min(stat_subset_size, len(data_loader)),
                    "shuffle_data": True,
                },
            }
        ]

        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

        pipeline = create_pipeline(algorithms, engine)

        compressed_model = pipeline.run(model)

        compress_model_weights(compressed_model)

        if optimization_parameters is not None:
            optimization_parameters.update_progress(90)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            with open(os.path.join(tempdir, "model.xml"), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            output_model.set_data(
                "confidence_threshold",
                np.array([self.confidence_threshold], dtype=np.float32).tobytes(),
            )

        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self.task_environment.label_schema),
        )
        output_model.set_data("config.json", config_to_bytes(self.hparams))

        # set model attributes for quantized model
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()
        logger.info("POT optimization completed")

        if optimization_parameters is not None:
            optimization_parameters.update_progress(100)
