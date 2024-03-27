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
import pickle
import random
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Type, Union
from zipfile import ZipFile

import attr
import cv2
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
                **attr.asdict(
                    hparams.postprocessing,
                    filter=lambda attr, value: attr.name in ["image_size", "resize_type", "downsizing"],
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
                        "downsizing",
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

    def pre_process(
        self,
        dataset_item: DatasetItemEntity,
        extra_processing: bool = False,
        use_bbox: bool = False,
        use_point: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """Pre-process function of OpenVINO Visual Prompting Inferencer for image encoder."""
        if use_bbox and use_point:
            logger.warning("If both use_bbox and use_point are set, bboxes and points will be generated randomly.")

        prob = 1.0 if not use_point else 0.0 if not use_bbox and use_point else 0.5
        images, meta = self.model["image_encoder"].preprocess(dataset_item.numpy, extra_processing)
        prompts = OTXVisualPromptingDataset.get_prompts(dataset_item, self.labels, prob=prob)
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
            prompt.update(image_embeddings)

            # forward decoder to get predicted mask
            prediction = self.forward_decoder(prompt)
            prediction["scores"] = prediction["iou_predictions"]
            metadata = {"label": label}

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
        super().__init__(hparams, label_schema, model_files, weight_files, device, num_requests)

        self.point_labels_box = np.array([[2, 3]], dtype=np.float32)
        self.has_mask_inputs = [np.array([[0.0]]), np.array([[1.0]])]

        self.reference_feats: Optional[np.ndarray] = None
        self.used_indices: Optional[np.ndarray] = None

    def pre_process_image_encoder(
        self, inputs: np.ndarray, extra_processing: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Pre-process function of OpenVINO Zero-shot Visual Prompting Inferencer for image encoder."""
        return self.model["image_encoder"].preprocess(inputs, extra_processing)

    def learn(
        self,
        dataset_item: DatasetItemEntity,
        reset_feat: bool = False,
        use_bbox: bool = False,
        use_point: bool = False,
        path_reference_info: str = "vpm_zsl_reference_infos/{}/reference_info.pickle",
        default_threshold_reference: float = 0.3,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Learn for reference features."""
        ref_masks: np.ndarray

        if reset_feat or self.reference_feats is None:
            self.initialize_reference_info()

        images, meta, prompts = self.pre_process(dataset_item, use_bbox, use_point)
        largest_label: int = max([int(p["label"].id) for p in prompts])
        self.expand_reference_info(largest_label)

        image_embeddings = self.forward_image_encoder(images)
        processed_embedding = image_embeddings["image_embeddings"].squeeze().transpose(1, 2, 0)
        original_size = meta["original_shape"][:2]

        ref_masks = np.zeros((largest_label + 1, *map(int, original_size)), dtype=np.uint8)
        for prompt in prompts:
            if "point_coords" in prompt:
                # bboxes and points
                label = prompt.pop("label")
                original_size = prompt.get("orig_size")
                prompt.update(image_embeddings)

                prediction = self.forward_decoder(prompt, original_size, is_cascade=False)
                ref_mask = prediction["upscaled_masks"]
            else:
                logger.warning("annotation and polygon will be supported.")
                continue
            ref_masks[int(label.id)] += ref_mask

        ref_masks = np.clip(ref_masks, 0, 1)
        for label in range(largest_label + 1):
            ref_mask = ref_masks[label]
            if ref_mask.sum() == 0:
                # empty prediction
                continue

            ref_feat = None
            cur_default_threshold_reference = deepcopy(default_threshold_reference)
            while ref_feat is None:
                logger.info(f"[*] default_threshold_reference : {cur_default_threshold_reference:.4f}")
                ref_feat = self._generate_masked_features(
                    processed_embedding, ref_masks[label], cur_default_threshold_reference
                )
                cur_default_threshold_reference -= 0.05

            self.reference_feats[label] = ref_feat
            self.used_indices = np.concatenate((self.used_indices, np.array([label])))

        reference_info = {"reference_feats": self.reference_feats, "used_indices": self.used_indices}
        path_reference_info = path_reference_info.format(time.strftime("%Y%m%d-%H%M%S"))
        logger.info(f"Saved reference info at {path_reference_info}.")
        pickle.dump(reference_info, open(path_reference_info, "wb"))
        return reference_info, ref_masks

    def infer(
        self,
        images: np.ndarray,
        reference_feats: np.ndarray,
        used_indices: np.ndarray,
        is_cascade: bool = False,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        default_threshold_target: float = 0.65,
    ) -> Tuple[List[Any], DefaultDict[Any, Any], DefaultDict[Any, Any]]:
        """Perform a prediction for a given input image."""
        points_score: np.ndarray

        # forward image encoder
        images, meta = self.pre_process_image_encoder(images)
        original_shape = np.asarray(meta["original_shape"][:2], dtype=np.int64)
        image_embeddings = self.forward_image_encoder(images)

        # get point candidates
        total_points_scores, total_bg_coords = self._get_prompt_candidates(
            image_embeddings=image_embeddings["image_embeddings"],
            reference_feats=reference_feats,
            used_indices=used_indices,
            original_shape=original_shape,
            threshold=threshold,
            num_bg_points=num_bg_points,
            default_threshold_target=default_threshold_target,
            image_size=self.model["image_encoder"].image_size,
            downsizing=self.model["image_encoder"].downsizing,
        )

        annotations: DefaultDict = defaultdict(list)
        predicted_masks: DefaultDict = defaultdict(list)
        used_points: DefaultDict = defaultdict(list)
        for label in total_points_scores.keys():
            points_scores = total_points_scores[label]
            bg_coords = total_bg_coords[label]
            for points_score in points_scores:
                if points_score[-1] in [-1.0, 0.0]:
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
                point_coords = self.model["decoder"]._apply_coords(point_coords, original_shape)
                point_labels = np.array([1] + [0] * len(bg_coords), dtype=np.float32)
                inputs_decoder = {
                    "point_coords": point_coords[None],
                    "point_labels": point_labels[None],
                    "orig_size": original_shape[None],
                }
                inputs_decoder.update(image_embeddings)

                prediction = self.forward_decoder(inputs_decoder, original_shape, is_cascade)
                prediction.update({"scores": points_score[-1]})

                predicted_masks[label].append(prediction[self.model["decoder"].output_blob_name])
                used_points[label].append(points_score)

        self._inspect_overlapping_areas(predicted_masks, used_points)

        for label, predictions in predicted_masks.items():
            if len(predictions) == 0:
                continue
            metadata = {
                "label": [_label for _label in self.labels if int(_label.id_) == label][0],
                "original_size": original_shape,
            }
            for prediction, used_point in zip(predictions, used_points[label]):
                annotation, _, _ = self.post_process(
                    {self.model["decoder"].output_blob_name: prediction, "scores": used_point[-1]}, metadata
                )
                annotations[label].extend(annotation)

        return sum(annotations.values(), []), predicted_masks, used_points

    def forward_decoder(  # type: ignore
        self,
        inputs: Dict[str, np.ndarray],
        original_size: np.ndarray,
        is_cascade: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Visual Prompting Inferencer."""
        masks: np.ndarray
        logits: np.ndarray
        scores: np.ndarray
        num_iter = 3 if is_cascade else 1
        for i in range(num_iter):
            if i == 0:
                # First-step prediction
                mask_input = np.zeros(
                    (1, 1, *map(lambda x: x * 4, inputs["image_embeddings"].shape[2:])), dtype=np.float32
                )
                has_mask_input = self.has_mask_inputs[0]

            elif i == 1:
                # Cascaded Post-refinement-1
                mask_input, masks = self._postprocess_masks(masks, logits, scores, is_single=True)  # noqa: F821
                if masks.sum() == 0:
                    return {"upscaled_masks": masks}

                has_mask_input = self.has_mask_inputs[1]

            elif i == 2:
                # Cascaded Post-refinement-2
                mask_input, masks = self._postprocess_masks(masks, logits, scores)  # noqa: F821
                if masks.sum() == 0:
                    return {"upscaled_masks": masks}

                has_mask_input = self.has_mask_inputs[1]
                y, x = np.nonzero(masks)
                box_coords = self.model["decoder"]._apply_coords(
                    np.array([[[x.min(), y.min()], [x.max(), y.max()]]], dtype=np.float32), original_size
                )
                inputs.update(
                    {
                        "point_coords": np.concatenate((inputs["point_coords"], box_coords), axis=1),
                        "point_labels": np.concatenate((inputs["point_labels"], self.point_labels_box), axis=1),
                    }
                )

            inputs.update({"mask_input": mask_input, "has_mask_input": has_mask_input})
            prediction = self.model["decoder"].infer_sync(inputs)
            upscaled_masks, scores, logits = (
                prediction["upscaled_masks"],
                prediction["iou_predictions"],
                prediction["low_res_masks"],
            )
            masks = upscaled_masks > self.model["decoder"].mask_threshold

        _, masks = self._postprocess_masks(masks, logits, scores)
        return {"upscaled_masks": masks}

    def _get_prompt_candidates(
        self,
        image_embeddings: np.ndarray,
        reference_feats: np.ndarray,
        used_indices: np.ndarray,
        original_shape: np.ndarray,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        default_threshold_target: float = 0.65,
        image_size: int = 1024,
        downsizing: int = 64,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Get prompt candidates."""
        target_feat = image_embeddings.squeeze()
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = target_feat / np.linalg.norm(target_feat, axis=0, keepdims=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

        total_points_scores: Dict[int, np.ndarray] = {}
        total_bg_coords: Dict[int, np.ndarray] = {}
        for label in used_indices:
            sim = reference_feats[label] @ target_feat
            sim = sim.reshape(h_feat, w_feat)
            sim = self._resize_to_original_shape(sim, image_size, original_shape)

            threshold = (threshold == 0) * default_threshold_target + threshold
            points_scores, bg_coords = self._point_selection(
                mask_sim=sim,
                original_shape=original_shape,
                threshold=threshold,
                num_bg_points=num_bg_points,
                image_size=image_size,
                downsizing=downsizing,
            )

            if points_scores is not None:
                total_points_scores[label] = points_scores
                total_bg_coords[label] = bg_coords
        return total_points_scores, total_bg_coords

    def _point_selection(
        self,
        mask_sim: np.ndarray,
        original_shape: np.ndarray,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        image_size: int = 1024,
        downsizing: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select point used as point prompts."""
        _, w_sim = mask_sim.shape

        # Top-first point selection
        point_coords = np.where(mask_sim > threshold)
        fg_coords_scores = np.stack(point_coords[::-1] + (mask_sim[point_coords],), axis=0).T

        ## skip if there is no point coords
        if len(fg_coords_scores) == 0:
            return None, None

        ratio = image_size / original_shape.max()
        width = (original_shape[1] * ratio).astype(np.int64)
        n_w = width // downsizing

        ## get grid numbers
        idx_grid = fg_coords_scores[:, 1] * ratio // downsizing * n_w + fg_coords_scores[:, 0] * ratio // downsizing
        idx_grid_unique = np.unique(idx_grid.astype(np.int64))

        ## get matched indices
        matched_matrix = np.expand_dims(idx_grid, axis=-1) == idx_grid_unique  # (totalN, uniqueN)

        ## sample fg_coords_scores matched by matched_matrix
        matched_grid = np.expand_dims(fg_coords_scores, axis=1) * np.expand_dims(matched_matrix, axis=-1)

        ## sample the highest score one of the samples that are in the same grid
        matched_indices = self._topk_numpy(matched_grid[..., -1], k=1, axis=0, largest=True)[1][0].astype(np.int64)
        points_scores = matched_grid[matched_indices].diagonal().T

        ## sort by the highest score
        sorted_points_scores_indices = np.flip(np.argsort(points_scores[:, -1]), axis=-1).astype(np.int64)
        points_scores = points_scores[sorted_points_scores_indices]

        # Top-last point selection
        bg_indices = self._topk_numpy(mask_sim.flatten(), num_bg_points, largest=False)[1]
        bg_x = np.expand_dims(bg_indices // w_sim, axis=0)
        bg_y = bg_indices - bg_x * w_sim
        bg_coords = np.concatenate((bg_y, bg_x), axis=0).transpose(1, 0)
        bg_coords = bg_coords.astype(np.float32)

        return points_scores, bg_coords

    def _postprocess_masks(
        self, masks: np.ndarray, logits: np.ndarray, scores: np.ndarray, is_single: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """Post-process logits for resized masks according to best index based on scores."""
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
                return None, np.zeros(masks.shape[-2:])

            best_idx = np.argmax(scores[0])
        return logits[:, [best_idx]], masks[0, best_idx]

    def _resize_to_original_shape(self, masks: np.ndarray, image_size: int, original_shape: np.ndarray) -> np.ndarray:
        """Resize feature size to original shape."""
        # resize feature size to input size
        masks = cv2.resize(masks, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # remove pad
        prepadded_size = self._get_prepadded_size(original_shape, image_size)
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]

        # resize unpadded one to original shape
        original_shape = original_shape.astype(np.int64)
        h, w = original_shape[0], original_shape[1]
        return cv2.resize(masks, (w, h), interpolation=cv2.INTER_LINEAR)

    def _get_prepadded_size(self, original_shape: int, image_size: int) -> np.ndarray:
        """Get pre-padded size."""
        scale = image_size / np.max(original_shape)
        transformed_size = scale * original_shape
        return np.floor(transformed_size + 0.5).astype(np.int64)

    def _inspect_overlapping_areas(
        self,
        predicted_masks: Dict[int, List[np.ndarray]],
        used_points: Dict[int, List[np.ndarray]],
        threshold_iou: float = 0.8,
    ):
        def _calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray):
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
                _mask_iou = _calculate_mask_iou(mask, other_mask)
                if _mask_iou > threshold_iou:
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        overlapped_other_label.append(jm)
                    else:
                        overlapped_label.append(im)
                elif _mask_iou > 0:
                    # refine the slightly overlapping region
                    overlapped_coords = np.where(np.logical_and(mask, other_mask))
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        other_mask[overlapped_coords] = 0.0
                    else:
                        mask[overlapped_coords] = 0.0

            for im in sorted(list(set(overlapped_label)), reverse=True):
                masks.pop(im)
                used_points[label].pop(im)

            for jm in sorted(list(set(overlapped_other_label)), reverse=True):
                other_masks.pop(jm)
                used_points[other_label].pop(jm)

    def predict(self, dataset_item: DatasetItemEntity) -> List[Annotation]:  # type: ignore
        """Perform a prediction for a given input image."""
        results = self.infer(dataset_item.numpy, self.reference_feats, self.used_indices)
        return results[0]

    def _find_latest_reference_info(self, root: str = "vpm_zsl_reference_infos") -> Union[str, None]:
        """Find latest reference info to be used."""
        if not os.path.isdir(root):
            return None
        if len(stamps := sorted(os.listdir(root), reverse=True)) > 0:
            return stamps[0]
        return None

    def _get_reference_info(
        self, root: str = "vpm_zsl_reference_infos", path_reference_info: str = "{}/reference_info.pickle"
    ) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """Get reference info through loading previously saved one or running `learn`."""
        if (latest_stamp := self._find_latest_reference_info(root)) is not None:
            # load previously saved reference info
            latest_reference_info = os.path.join(root, path_reference_info.format(latest_stamp))
            reference_info = pickle.load(open(latest_reference_info, "rb"))
            return reference_info["reference_feats"], reference_info["used_indices"]
        return None, None

    def initialize_reference_info(self) -> None:
        """Initialize reference information."""
        self.reference_feats = np.zeros((0, 1, 256), dtype=np.float32)
        self.used_indices = np.array([], dtype=np.int64)

    def expand_reference_info(self, new_largest_label: int) -> None:
        """Expand reference info dimensions if newly given processed prompts have more lables."""
        if new_largest_label > (cur_largest_label := len(self.reference_feats) - 1):
            diff = new_largest_label - cur_largest_label
            self.reference_feats = np.pad(self.reference_feats, ((0, diff), (0, 0), (0, 0)), constant_values=0.0)

    def _generate_masked_features(
        self,
        feats: np.ndarray,
        masks: np.ndarray,
        threshold_mask: float,
    ) -> Tuple[np.ndarray, ...]:
        """Generate masked features.

        Args:
            feats (np.ndarray): Raw reference features. It will be filtered with masks.
            masks (np.ndarray): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.

        Returns:
            (np.ndarray): Masked features.
        """
        target_shape = self.model["image_encoder"].image_size / max(masks.shape) * np.array(masks.shape)
        target_shape = target_shape[::-1].astype(np.int32)

        # Post-process masks
        masks = cv2.resize(masks, target_shape, interpolation=cv2.INTER_LINEAR)
        masks = self._pad_to_square(masks)
        masks = cv2.resize(masks, feats.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None

        masked_feat = feats[masks > threshold_mask]
        masked_feat = masked_feat.mean(0)[None]
        masked_feat = masked_feat / np.linalg.norm(masked_feat, axis=-1, keepdims=True)

        return masked_feat

    def _pad_to_square(self, x: np.ndarray) -> np.ndarray:
        """Pad to a square input.

        Args:
            x (np.ndarray): Mask to be padded.

        Returns:
            (np.ndarray): Padded mask.
        """
        h, w = x.shape[-2:]
        padh = self.model["image_encoder"].image_size - h
        padw = self.model["image_encoder"].image_size - w
        x = np.pad(x, ((0, padh), (0, padw)), constant_values=0.0)
        return x

    def _topk_numpy(self, x: np.ndarray, k: int, axis: int = -1, largest: bool = True) -> np.ndarray:
        """Top-k function for numpy same with torch.topk."""
        if largest:
            k = -k
            indices = range(k, 0)
        else:
            indices = range(k)
        partitioned_ind = np.argpartition(x, k, axis=axis).take(indices=indices, axis=axis)
        partitioned_scores = np.take_along_axis(x, partitioned_ind, axis=axis)
        sorted_trunc_ind = np.flip(np.argsort(partitioned_scores, axis=axis), axis=axis)
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
        return scores, ind


class OTXOpenVinoDataLoader:
    """DataLoader implementation for VisualPromptingOpenVINOTask."""

    def __init__(
        self,
        dataset: Any,
        inferencer: OpenVINOVisualPromptingInferencer,
        module_name: str,
        shuffle: bool = True,
        output_model: Optional[ModelEntity] = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.inferencer = inferencer
        self.module_name = module_name
        self.shuffler = None
        if shuffle:
            self.shuffler = list(range(len(dataset)))
            random.shuffle(self.shuffler)

        self.target_length = self.inferencer.model["image_encoder"].orig_width
        if self.module_name not in ["image_encoder"]:
            self.image_encoder = self._load_module("image_encoder", output_model)

    def _load_module(self, module_name: str, output_model: ModelEntity, core=ov.Core()):
        """Load specific module."""
        compressed_model = core.read_model(
            output_model.get_data(f"visual_prompting_{module_name}.xml"),
            output_model.get_data(f"visual_prompting_{module_name}.bin"),
        )
        return core.compile_model(
            model=compressed_model, device_name=self.inferencer.model[module_name].inference_adapter.device
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
        if self.module_name == "image_encoder":
            return images
        else:
            image_embeddings = self.image_encoder(images["images"])
            prompt = prompts[0]  # only use the first prompt
            prompt.pop("label")
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
        parameters: Dict[str, Any] = {}
        parameters["converter_type"] = f"{self.task_type}"
        parameters["model_parameters"] = self.inferencer.configuration
        parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(self.task_environment.label_schema)

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
        module_names: List[str] = ["image_encoder", "decoder"],
        ov_dataloader: Type[OTXOpenVinoDataLoader] = OTXOpenVinoDataLoader,
        **kwargs,
    ):
        """Optimize function of OpenVINOVisualPromptingTask."""
        logger.info("Start PTQ optimization")
        if self.model is None:
            raise RuntimeError("PTQ optimize failed, model is None")

        if optimization_type is not OptimizationType.POT:
            raise ValueError("PTQ is the only supported optimization type for OpenVino models")

        dataset = dataset.get_subset(Subset.TRAINING)

        for i, module_name in enumerate(module_names, 1):
            data_loader = ov_dataloader(
                dataset, self.inferencer, module_name=module_name, output_model=output_model, **kwargs
            )
            quantization_dataset = nncf.Dataset(data_loader, lambda data: data)

            with tempfile.TemporaryDirectory() as tempdir:
                xml_path = os.path.join(tempdir, f"visual_prompting_{module_name}.xml")
                bin_path = os.path.join(tempdir, f"visual_prompting_{module_name}.bin")
                with open(xml_path, "wb") as f:
                    f.write(self.model.get_data(f"visual_prompting_{module_name}.xml"))
                with open(bin_path, "wb") as f:
                    f.write(self.model.get_data(f"visual_prompting_{module_name}.bin"))

                ov_model = ov.Core().read_model(xml_path, bin_path)
                if check_if_quantized(ov_model):
                    raise RuntimeError("Model is already optimized by PTQ")

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
                optimization_parameters.update_progress(90 // len(module_names) * i, None)

            with tempfile.TemporaryDirectory() as tempdir:
                xml_path = os.path.join(tempdir, f"visual_prompting_{module_name}.xml")
                bin_path = os.path.join(tempdir, f"visual_prompting_{module_name}.bin")
                ov.save_model(compressed_model, xml_path)
                with open(xml_path, "rb") as f:
                    output_model.set_data(f"visual_prompting_{module_name}.xml", f.read())
                with open(bin_path, "rb") as f:
                    output_model.set_data(f"visual_prompting_{module_name}.bin", f.read())

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
                "decoder": self.model.get_data("visual_prompting_decoder.xml"),
            },
            weight_files={
                "image_encoder": self.model.get_data("visual_prompting_image_encoder.bin"),
                "decoder": self.model.get_data("visual_prompting_decoder.bin"),
            },
            num_requests=get_default_async_reqs_num(),
        )

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
        root: str = "vpm_zsl_reference_infos",
        path_reference_info: str = "{}/reference_info.pickle",
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

        if self.inferencer.reference_feats is None and self.inferencer.used_indices is None:
            # set reference_feats and used_indices from previously saved reference_info
            self.inferencer.reference_feats, self.inferencer.used_indices = self.inferencer._get_reference_info(
                root, path_reference_info
            )
            if self.inferencer.reference_feats is None and self.inferencer.used_indices is None:
                # if they are empty, stop inference and return empty dataset
                logger.warning(
                    (
                        "reference_feats and used_indices are empty, stop inference and return empty dataset. "
                        "Please run learn function first."
                    )
                )
                return predicted_validation_dataset

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

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
        module_names: List[str] = ["image_encoder", "decoder"],
        ov_dataloader: Type[OTXOpenVinoDataLoader] = OTXOpenVinoDataLoader,
        **kwargs,
    ):
        """Optimize function of OpenVINOZeroShotVisualPromptingTask."""
        self.inferencer: OpenVINOZeroShotVisualPromptingInferencer
        reference_feats, used_indices = self.inferencer._get_reference_info()
        return super().optimize(
            optimization_type=optimization_type,
            dataset=dataset,
            output_model=output_model,
            optimization_parameters=optimization_parameters,
            module_names=module_names,
            ov_dataloader=ov_dataloader,
            reference_feats=reference_feats,
            used_indices=used_indices,
        )
