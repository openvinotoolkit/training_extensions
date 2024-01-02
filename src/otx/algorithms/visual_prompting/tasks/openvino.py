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
import random
import tempfile
import time
from itertools import product
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import attr
import nncf
import numpy as np
import openvino.runtime as ov
from addict import Dict as ADDict
from nncf.common.quantization.structs import QuantizationPreset
from openvino.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_api.models import Model

from otx.algorithms.common.utils import get_default_async_reqs_num, read_py_config
from otx.algorithms.common.utils.ir import check_if_quantized
from otx.algorithms.visual_prompting.adapters.openvino import model_wrappers
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.dataset import (
    OTXVisualPromptingDataset,
    get_transform,
)
from otx.algorithms.visual_prompting.configs.base import VisualPromptingBaseConfig
from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.model_template import TaskType
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.exportable_code import demo
from otx.api.usecases.exportable_code.inference.inference import IInferencer
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    VisualPromptingToAnnotationConverter,
)
from otx.api.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from otx.utils.logger import get_logger

logger = get_logger()


class OpenVINOVisualPromptingInferencer(IInferencer):
    """Inferencer implementation for Visual Prompting using OpenVINO backend.

    This inferencer has two models, image encoder and decoder.

    Args:
        hparams (VisualPromptingBaseConfig): Hyper parameters that the model should use.
        label_schema (LabelSchemaEntity): LabelSchemaEntity that was used during model training.
        model_files (Dict[str, Union[str, Path, bytes]]): Path or bytes to model to load,
            `.xml`, `.bin` or `.onnx` file.
        weight_files (Dict[str, Union[str, Path, bytes, None]], optional): Path or bytes to weights to load,
            `.xml`, `.bin` or `.onnx` file. Defaults to None.
        device (str): Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        num_requests (int) : Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
    """

    def __init__(
        self,
        hparams: VisualPromptingBaseConfig,
        label_schema: LabelSchemaEntity,
        model_files: Dict[str, Union[str, Path, bytes]],
        weight_files: Optional[Dict[str, Union[str, Path, bytes, None]]] = {},
        device: str = "CPU",
        num_requests: int = 1,
    ):

        assert all(module in model_files for module in ["image_encoder", "decoder"])

        self.model = {}
        model_parameters = {"decoder": {"input_layouts": "image_embeddings:NCHW"}}
        self.configuration = {
            "image_encoder": {
                **attr.asdict(hparams.postprocessing, filter=lambda attr, value: attr.name in ["image_size"])
            },
            "decoder": {
                **attr.asdict(
                    hparams.postprocessing,
                    filter=lambda attr, value: attr.name
                    not in [
                        "header",
                        "description",
                        "type",
                        "visible_in_ui",
                        "class_name",
                        "sim_threshold",
                        "num_bg_points",
                    ],
                )
            },
        }
        for name in ["image_encoder", "decoder"]:
            model_adapter = OpenvinoAdapter(
                core=create_core(),
                model=model_files.get(name),
                weights_path=weight_files.get(name, None),
                model_parameters=model_parameters.get(name, {}),
                device=device,
                max_num_requests=num_requests,
                plugin_config={"PERFORMANCE_HINT": "THROUGHPUT"},
            )
            self.model[name] = Model.create_model(model_adapter, name, self.configuration.get(name, {}), preload=True)
        self.converter = VisualPromptingToAnnotationConverter()
        self.labels = label_schema.get_labels(include_empty=False)
        self.transform = get_transform()  # TODO (sungchul): insert args

    def pre_process(  # type: ignore
        self, dataset_item: DatasetItemEntity, extra_processing: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """Pre-process function of OpenVINO Visual Prompting Inferencer for image encoder."""
        images, meta = self.model["image_encoder"].preprocess(dataset_item.numpy, extra_processing)
        prompts = OTXVisualPromptingDataset.get_prompts(dataset_item, self.labels)  # to be replaced
        prompts = self.model["decoder"].preprocess(prompts, meta)
        return images, meta, prompts  # type: ignore

    def post_process(
        self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Tuple[List[Annotation], Any, Any]:
        """Post-process function of OpenVINO Visual Prompting Inferencer."""
        hard_prediction, soft_prediction = self.model["decoder"].postprocess(prediction, metadata)
        annotation = self.converter.convert_to_annotation(hard_prediction, metadata)
        return annotation, hard_prediction, soft_prediction

    def predict(self, dataset_item: DatasetItemEntity) -> List[Annotation]:  # type: ignore
        """Perform a prediction for a given input image."""
        # forward image encoder
        images, meta, prompts = self.pre_process(dataset_item)
        image_embeddings = self.forward_image_encoder(images)

        annotations: List[Annotation] = []
        hard_predictions: List[np.ndarray] = []
        soft_predictions: List[np.ndarray] = []
        for prompt in prompts:
            label = prompt.pop("label")
            orig_size = prompt.pop("orig_size")
            prompt.update(image_embeddings)

            # forward decoder to get predicted mask
            prediction = self.forward_decoder(prompt)
            metadata = {"label": label, "original_size": orig_size}

            # set annotation for eval
            annotation, hard_prediction, soft_prediction = self.post_process(prediction, metadata)
            annotations.extend(annotation)
            hard_predictions.append(hard_prediction)
            soft_predictions.append(soft_prediction)
        return annotations

    def forward_image_encoder(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Visual Prompting Inferencer."""
        return self.model["image_encoder"].infer_sync(inputs)

    def forward_decoder(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Visual Prompting Inferencer."""
        return self.model["decoder"].infer_sync(inputs)

    def await_all(self) -> None:
        """Await all running infer requests if any."""
        self.model["image_encoder"].await_all()
        self.model["decoder"].await_all()


class OpenVINOZeroShotVisualPromptingInferencer(OpenVINOVisualPromptingInferencer):
    """Inferencer implementation for Zero-shot Visual Prompting using OpenVINO backend.

    This inferencer has two models, image encoder and decoder.

    Args:
        hparams (VisualPromptingBaseConfig): Hyper parameters that the model should use.
        label_schema (LabelSchemaEntity): LabelSchemaEntity that was used during model training.
        model_files (Dict[str, Union[str, Path, bytes]]): Path or bytes to model to load,
            `.xml`, `.bin` or `.onnx` file.
        weight_files (Dict[str, Union[str, Path, bytes, None]], optional): Path or bytes to weights to load,
            `.xml`, `.bin` or `.onnx` file. Defaults to None.
        device (str): Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        num_requests (int) : Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
    """

    def __init__(
        self,
        hparams: VisualPromptingBaseConfig,
        label_schema: LabelSchemaEntity,
        model_files: Dict[str, Union[str, Path, bytes]],
        weight_files: Optional[Dict[str, Union[str, Path, bytes, None]]] = {},
        device: str = "CPU",
        num_requests: int = 1,
    ):

        assert all(module in model_files for module in ["image_encoder", "prompt_getter", "decoder"])

        self.model = {}
        model_parameters = {
            "prompt_getter": {"input_layouts": "image_embeddings:NCHW"},
            "decoder": {"input_layouts": "image_embeddings:NCHW"},
        }
        self.configuration = {
            "image_encoder": {
                **attr.asdict(hparams.postprocessing, filter=lambda attr, value: attr.name in ["image_size"])
            },
            "prompt_getter": {
                **attr.asdict(
                    hparams.postprocessing,
                    filter=lambda attr, value: attr.name
                    in ["image_size", "sim_threshold", "num_bg_points", "embedded_processing"],
                )
            },
            "decoder": {
                **attr.asdict(
                    hparams.postprocessing,
                    filter=lambda attr, value: attr.name
                    not in [
                        "header",
                        "description",
                        "type",
                        "visible_in_ui",
                        "class_name",
                        "sim_threshold",
                        "num_bg_points",
                    ],
                )
            },
        }

        core = create_core()
        for name in ["image_encoder", "prompt_getter", "decoder"]:
            model_adapter = OpenvinoAdapter(
                core=core,
                model=model_files.get(name),
                weights_path=weight_files.get(name, None),
                model_parameters=model_parameters.get(name, {}),
                device=device,
                max_num_requests=num_requests,
                plugin_config={"PERFORMANCE_HINT": "THROUGHPUT"},
            )
            self.model[name] = Model.create_model(model_adapter, name, self.configuration.get(name, {}), preload=True)
        self.converter = VisualPromptingToAnnotationConverter()
        self.labels = label_schema.get_labels(include_empty=False)
        self.transform = get_transform()  # TODO (sungchul): insert args

        self.point_labels_box = np.array([[2, 3]], dtype=np.float32)
        self.has_mask_inputs = [np.array([[0.0]]), np.array([[1.0]])]

    def pre_process(  # type: ignore
        self, dataset_item: DatasetItemEntity, extra_processing: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Pre-process function of OpenVINO Zero-shot Visual Prompting Inferencer for image encoder."""
        return self.model["image_encoder"].preprocess(dataset_item.numpy, extra_processing)

    def pre_process_prompt_getter(
        self, image_embeddings: Dict[str, np.ndarray], original_size: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Pre-process function of OpenVINO Zero-shot VIsual Prompting Inferencer for prompt getter."""
        inputs_prompt_getter = {
            "original_size": original_size[None],
            "threshold": np.array([[self.model["prompt_getter"].sim_threshold]], dtype=np.float32),
            "num_bg_points": np.array([[self.model["prompt_getter"].num_bg_points]], dtype=np.int64),
        }
        inputs_prompt_getter.update(image_embeddings)
        return inputs_prompt_getter

    def predict(self, dataset_item: DatasetItemEntity) -> List[Annotation]:  # type: ignore
        """Perform a prediction for a given input image."""
        # forward image encoder
        images, meta = self.pre_process(dataset_item)
        original_size = np.array(meta["original_shape"][:2], dtype=np.int64)
        image_embeddings = self.forward_image_encoder(images)

        # get point candidates
        inputs_prompt_getter = self.pre_process_prompt_getter(image_embeddings, original_size)
        total_prompts = self.forward_prompt_getter(inputs_prompt_getter)

        annotations: DefaultDict = defaultdict(list)
        predicted_masks: DefaultDict = defaultdict(list)
        used_points: DefaultDict = defaultdict(list)
        for label, (points_scores, bg_coords) in enumerate(
            zip(total_prompts["total_points_scores"], total_prompts["total_bg_coords"])
        ):
            for points_score in points_scores:
                if points_score[-1] == -1:
                    continue
                x, y = points_score[:2]
                is_done = False
                for pm in predicted_masks.get(label, []):
                    # check if that point is already assigned
                    if pm[int(y), int(x)] > 0:
                        is_done = True
                        break
                if is_done:
                    continue

                point_coords = np.concatenate((np.array([[x, y]]), bg_coords), axis=0, dtype=np.float32)
                point_coords = self.model["decoder"]._apply_coords(point_coords, original_size)
                point_labels = np.array([1] + [0] * len(bg_coords), dtype=np.float32)
                inputs_decoder = {"point_coords": point_coords[None], "point_labels": point_labels[None]}
                inputs_decoder.update(image_embeddings)

                prediction = self.forward_decoder(inputs_decoder, original_size)
                metadata = {
                    "label": [_label for _label in self.labels if int(_label.id_) == label][0],
                    "original_size": original_size[None],
                }

                # set annotation for eval
                annotation, hard_prediction, _ = self.post_process(prediction, metadata)
                annotations[label].extend(annotation)
                predicted_masks[label].append(hard_prediction)
                used_points[label].append(points_score)
        self.__inspect_overlapping_areas(predicted_masks, used_points, annotations)
        return sum(annotations.values(), [])

    def forward_prompt_getter(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Visual Prompting Inferencer."""
        return self.model["prompt_getter"].infer_sync(inputs)

    def forward_decoder(  # type: ignore
        self, inputs: Dict[str, np.ndarray], original_size: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Visual Prompting Inferencer."""
        logits: np.ndarray
        scores: np.ndarray
        mask_slice = slice(0, 1)
        for i in range(3):
            if i == 0:
                # First-step prediction
                mask_input = np.zeros(
                    (1, 1, *map(lambda x: x * 4, inputs["image_embeddings"].shape[2:])), dtype=np.float32
                )
                has_mask_input = self.has_mask_inputs[0]

            elif i == 1:
                # Cascaded Post-refinement-1
                mask_input, masks, iou_predictions = self._postprocess_masks(
                    logits, scores, original_size, is_single=True  # noqa: F821
                )
                if masks.sum() == 0:
                    return {"iou_predictions": iou_predictions, "low_res_masks": mask_input}

                has_mask_input = self.has_mask_inputs[1]

            elif i == 2:
                # Cascaded Post-refinement-2
                mask_input, masks, iou_predictions = self._postprocess_masks(
                    logits, scores, original_size  # noqa: F821
                )
                if masks.sum() == 0:
                    return {"iou_predictions": iou_predictions, "low_res_masks": mask_input}

                has_mask_input = self.has_mask_inputs[1]
                y, x = np.nonzero(masks)
                box_coords = self.model["decoder"]._apply_coords(
                    np.array([[[x.min(), y.min()], [x.max(), y.max()]]], dtype=np.float32), original_size
                )
                inputs["point_coords"] = np.concatenate((inputs["point_coords"], box_coords), axis=1)
                inputs["point_labels"] = np.concatenate((inputs["point_labels"], self.point_labels_box), axis=1)

            inputs.update({"mask_input": mask_input, "has_mask_input": has_mask_input})
            prediction = self.model["decoder"].infer_sync(inputs)
            scores, logits = prediction["iou_predictions"], prediction["low_res_masks"]

        return {"iou_predictions": scores[:, mask_slice], "low_res_masks": logits[:, mask_slice, :, :]}

    def _postprocess_masks(
        self, logits: np.ndarray, scores: np.ndarray, original_size: np.ndarray, is_single: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """Post-process logits for resized masks according to best index based on scores."""
        high_res_masks = self.model["decoder"].resize_and_crop(logits[0].transpose(1, 2, 0), original_size)
        masks = high_res_masks > self.model["decoder"].mask_threshold
        masks = masks.transpose(2, 0, 1)[None]

        if is_single:
            best_idx = 0
        else:
            # skip the first index components
            scores, masks, logits = map(lambda x: x[:, 1:], (scores, masks, logits))

            # filter zero masks
            while len(scores[0]) > 0 and masks[0, (best_idx := np.argmax(scores[0]))].sum() == 0:
                scores, masks, logits = map(
                    lambda x: np.concatenate((x[:, :best_idx], x[:, best_idx + 1 :]), axis=1), (scores, masks, logits)
                )

            if len(scores[0]) == 0:
                # all predicted masks were zero masks, ignore them.
                return None, np.zeros((self.model["decoder"].image_size, self.model["decoder"].image_size)), 0.0

            best_idx = np.argmax(scores[0])
        return logits[:, [best_idx]], masks[0, best_idx], scores[0, best_idx]
    
    def __inspect_overlapping_areas(self, predicted_masks: Dict[int, List[np.ndarray]], used_points: Dict[int, List[np.ndarray]], annotations: Dict[int, List[np.ndarray]], threshold_iou: float = 0.8):
        def __calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray):
            assert mask1.ndim == 2 and mask2.ndim == 2
            intersection = np.logical_and(mask1, mask2).sum().item()
            union = np.logical_or(mask1, mask2).sum().item()

            # Avoid division by zero
            if union == 0:
                return 0.0
            iou = intersection / union
            return iou

        for (label, masks), (other_label, other_masks) in product(predicted_masks.items(), predicted_masks.items()):
            if other_label <= label:
                continue

            overlapped_label = []
            overlapped_other_label = []
            for (im, mask), (jm, other_mask) in product(enumerate(masks), enumerate(other_masks)):
                if __calculate_mask_iou(mask, other_mask) > threshold_iou:
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        overlapped_other_label.append(jm)
                    else:
                        overlapped_label.append(im)

            for im in overlapped_label[::-1]:
                masks.pop(im)
                used_points[label].pop(im)
                annotations[label].pop(im)

            for jm in overlapped_other_label[::-1]:
                other_masks.pop(jm)
                used_points[other_label].pop(jm)
                annotations[other_label].pop(jm)


class OTXOpenVinoDataLoader:
    """DataLoader implementation for VisualPromptingOpenVINOTask."""

    def __init__(
        self,
        dataset: Any,
        inferencer: OpenVINOVisualPromptingInferencer,
        shuffle: bool = True,
        is_encoder: bool = True,
        output_model: Optional[ModelEntity] = None,
    ):
        self.dataset = dataset
        self.inferencer = inferencer
        self.shuffler = None
        if shuffle:
            self.shuffler = list(range(len(dataset)))
            random.shuffle(self.shuffler)

        self.is_encoder = is_encoder
        self.target_length = self.inferencer.model["image_encoder"].orig_width
        if not self.is_encoder:
            core = ov.Core()
            compressed_model = core.read_model(
                output_model.get_data("visual_prompting_image_encoder.xml"),
                output_model.get_data("visual_prompting_image_encoder.bin"),
            )
            self.compressed_model = core.compile_model(
                model=compressed_model, device_name=inferencer.model["image_encoder"].inference_adapter.device
            )

    def __getitem__(self, index: int):
        """Get item from dataset."""
        if self.shuffler is not None:
            index = self.shuffler[index]

        items = self.dataset[index]
        images, _, prompts = self.inferencer.pre_process(items, extra_processing=True)
        _, _, h, w = images["images"].shape
        pad_width = ((0, 0), (0, 0), (0, self.target_length - h), (0, self.target_length - w))
        images["images"] = np.pad(images["images"], pad_width, mode="constant", constant_values=0)
        if self.is_encoder:
            return images
        else:
            image_embeddings = self.compressed_model(images["images"])
            prompt = prompts[0]  # only use the first prompt
            prompt.pop("label")
            prompt.pop("orig_size")
            prompt.update({"image_embeddings": image_embeddings["image_embeddings"]})
            return prompt
            # TODO (sungchul): change has_mask_input

    def __len__(self):
        """Get length of dataset."""
        return len(self.dataset)


class OpenVINOVisualPromptingTask(IInferenceTask, IEvaluationTask, IOptimizationTask, IDeploymentTask):
    """Task implementation for Visual Prompting using OpenVINO backend."""

    def __init__(self, task_environment: TaskEnvironment) -> None:
        self.task_environment = task_environment
        self.model = self.task_environment.model
        self.model_name = self.task_environment.model_template.model_template_id
        self.inferencer = self.load_inferencer()
        self._avg_time_per_image: Optional[float] = None

        labels = task_environment.get_labels(include_empty=False)
        self._label_dictionary = dict(enumerate(labels, 1))
        template_file_path = self.task_environment.model_template.model_template_path
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))
        self.task_type = TaskType.VISUAL_PROMPTING

    @property
    def hparams(self):
        """Hparams of OpenVINO Visual Prompting Task."""
        return self.task_environment.get_hyper_parameters(VisualPromptingBaseConfig)

    @property
    def avg_time_per_image(self) -> Optional[float]:
        """Average inference time per image."""
        return self._avg_time_per_image

    def load_inferencer(self) -> OpenVINOVisualPromptingInferencer:
        """Load OpenVINO Visual Prompting Inferencer."""
        if self.model is None:
            raise RuntimeError("load_inferencer failed, model is None")
        return OpenVINOVisualPromptingInferencer(
            self.hparams,
            self.task_environment.label_schema,
            {
                "image_encoder": self.model.get_data("visual_prompting_image_encoder.xml"),
                "decoder": self.model.get_data("visual_prompting_decoder.xml"),
            },
            {
                "image_encoder": self.model.get_data("visual_prompting_image_encoder.bin"),
                "decoder": self.model.get_data("visual_prompting_decoder.bin"),
            },
            num_requests=get_default_async_reqs_num(),
        )

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Infer function of OpenVINOVisualPromptingTask.

        Currently, asynchronous execution is not supported, synchronous execution will be executed instead.
        """
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            enable_async_inference = inference_parameters.enable_async_inference
        else:
            update_progress_callback = default_progress_callback
            enable_async_inference = True

        # FIXME (sungchul): Support async inference.
        if enable_async_inference:
            logger.warning("Asynchronous inference doesn't work, synchronous inference will be executed.")
            enable_async_inference = False
        predicted_validation_dataset = dataset.with_empty_annotations()

        def add_prediction(id: int, annotations: List[Annotation]):
            dataset_item = predicted_validation_dataset[id]
            dataset_item.append_annotations(annotations)

        total_time = 0.0
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            start_time = time.perf_counter()

            annotations = self.inferencer.predict(dataset_item)
            add_prediction(i - 1, annotations)

            end_time = time.perf_counter() - start_time
            total_time += end_time
            update_progress_callback(int(i / dataset_size * 100), None)

        self.inferencer.await_all()

        self._avg_time_per_image = total_time / len(dataset)
        logger.info(f"Avg time per image: {self._avg_time_per_image} secs")
        logger.info(f"Total time: {total_time} secs")
        logger.info("Visual Prompting OpenVINO inference completed")

        return predicted_validation_dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate function of OpenVINOVisualPromptingTask."""
        logger.info("Computing mDice")
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")

        output_resultset.performance = metrics.get_performance()

    def deploy(self, output_model: ModelEntity) -> None:
        """Deploy function of OpenVINOVisualPromptingTask."""
        logger.info("Deploying the model")
        if self.model is None:
            raise RuntimeError("deploy failed, model is None")

        work_dir = os.path.dirname(demo.__file__)
        parameters = {}
        parameters["converter_type"] = f"{self.task_type}"
        parameters["model_parameters"] = self.inferencer.configuration  # type: ignore
        parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(self.task_environment.label_schema)  # type: ignore # noqa: E501

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            arch.writestr(
                os.path.join("model", "visual_prompting_image_encoder.xml"),
                self.model.get_data("visual_prompting_image_encoder.xml"),
            )
            arch.writestr(
                os.path.join("model", "visual_prompting_image_encoder.bin"),
                self.model.get_data("visual_prompting_image_encoder.bin"),
            )
            arch.writestr(
                os.path.join("model", "visual_prompting_decoder.xml"),
                self.model.get_data("visual_prompting_decoder.xml"),
            )
            arch.writestr(
                os.path.join("model", "visual_prompting_decoder.bin"),
                self.model.get_data("visual_prompting_decoder.bin"),
            )
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
                            file_path.split("model_wrappers/")[0],
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
        """Optimize function of OpenVINOVisualPromptingTask."""
        logger.info("Start PTQ optimization")
        if self.model is None:
            raise RuntimeError("PTQ optimize failed, model is None")

        if optimization_type is not OptimizationType.POT:
            raise ValueError("PTQ is the only supported optimization type for OpenVino models")

        dataset = dataset.get_subset(Subset.TRAINING)

        for i, (name, is_encoder) in enumerate(zip(["image_encoder", "decoder"], [True, False]), 1):
            data_loader = OTXOpenVinoDataLoader(
                dataset, self.inferencer, is_encoder=is_encoder, output_model=output_model
            )
            quantization_dataset = nncf.Dataset(data_loader, lambda data: data)

            with tempfile.TemporaryDirectory() as tempdir:
                xml_path = os.path.join(tempdir, f"visual_prompting_{name}.xml")
                bin_path = os.path.join(tempdir, f"visual_prompting_{name}.bin")
                with open(xml_path, "wb") as f:
                    f.write(self.model.get_data(f"visual_prompting_{name}.xml"))
                with open(bin_path, "wb") as f:
                    f.write(self.model.get_data(f"visual_prompting_{name}.bin"))

                ov_model = ov.Core().read_model(xml_path, bin_path)
                if check_if_quantized(ov_model):
                    raise RuntimeError("Model is already optimized by PTQ")

            if optimization_parameters is not None:
                optimization_parameters.update_progress(10 * i + 35 * (i - 1), None)

            optimization_config_path = os.path.join(self._base_dir, "ptq_optimization_config.py")
            ptq_config = ADDict()
            if os.path.exists(optimization_config_path):
                ptq_config = read_py_config(optimization_config_path)
            ptq_config.update(
                subset_size=min(self.hparams.pot_parameters.stat_subset_size, len(data_loader)),
                preset=QuantizationPreset(self.hparams.pot_parameters.preset.name.lower()),
            )

            compressed_model = nncf.quantize(ov_model, quantization_dataset, **ptq_config)

            if optimization_parameters is not None:
                optimization_parameters.update_progress(45 * i, None)

            with tempfile.TemporaryDirectory() as tempdir:
                xml_path = os.path.join(tempdir, f"visual_prompting_{name}.xml")
                bin_path = os.path.join(tempdir, f"visual_prompting_{name}.bin")
                ov.serialize(compressed_model, xml_path)
                with open(xml_path, "rb") as f:
                    output_model.set_data(f"visual_prompting_{name}.xml", f.read())
                with open(bin_path, "rb") as f:
                    output_model.set_data(f"visual_prompting_{name}.bin", f.read())

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
        logger.info("PTQ optimization completed")


class OpenVINOZeroShotVisualPromptingTask(OpenVINOVisualPromptingTask):
    """Task implementation for Zero-shot Visual Prompting using OpenVINO backend."""

    def load_inferencer(self) -> OpenVINOZeroShotVisualPromptingInferencer:
        """Load OpenVINO Zero-shot Visual Prompting Inferencer."""
        if self.model is None:
            raise RuntimeError("load_inferencer failed, model is None")
        return OpenVINOZeroShotVisualPromptingInferencer(
            self.hparams,
            self.task_environment.label_schema,
            model_files={
                "image_encoder": self.model.get_data("visual_prompting_image_encoder.xml"),
                "prompt_getter": self.model.get_data("visual_prompting_prompt_getter.xml"),
                "decoder": self.model.get_data("visual_prompting_decoder.xml"),
            },
            weight_files={
                "image_encoder": self.model.get_data("visual_prompting_image_encoder.bin"),
                "prompt_getter": self.model.get_data("visual_prompting_prompt_getter.bin"),
                "decoder": self.model.get_data("visual_prompting_decoder.bin"),
            },
            num_requests=get_default_async_reqs_num(),
        )
