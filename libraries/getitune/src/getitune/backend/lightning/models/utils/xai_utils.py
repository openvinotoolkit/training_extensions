# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utils used for XAI."""

# TODO(gzalessk): Typings in this file is too weak or wrong. It should be fixed.
# For example, `pred_labels: list | None` has no object typing containered in the list.
# On the other hand, process_saliency_maps should not produce list of dictionaries
# (`list[dict[str, np.ndarray | torch.Tensor]]`).
# This is because the output will be assigned to PredictionBatch.saliency_map,
# but this has `list[np.ndarray | torch.Tensor]` typing, so that it makes a mismatch.

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from getitune.config.explain import ExplainConfig
from getitune.data.entity.sample import PredictionBatch
from getitune.types.explain import TargetExplainGroup

ProcessedSaliencyMaps = list[dict[str, np.ndarray | torch.Tensor]]


def process_saliency_maps_in_pred_entity(
    predict_result: list[PredictionBatch],
    explain_config: ExplainConfig,
) -> list[PredictionBatch]:
    """Process saliency maps in PredEntity."""
    processed_predict_result = []
    for predict_result_per_batch in predict_result:
        if predict_result_per_batch.saliency_map is None or len(predict_result_per_batch.saliency_map) == 0:
            continue

        # Extract batch data with proper type handling
        labels = predict_result_per_batch.labels if predict_result_per_batch.labels is not None else []
        scores = predict_result_per_batch.scores if predict_result_per_batch.scores is not None else []

        saliency_map: list[np.ndarray] = [
            saliency_map.cpu().numpy() if isinstance(saliency_map, torch.Tensor) else saliency_map
            for saliency_map in predict_result_per_batch.saliency_map
        ]
        imgs_info = predict_result_per_batch.imgs_info
        ori_img_shapes = [img_info.ori_shape for img_info in imgs_info]  # type: ignore[union-attr]
        paddings = [img_info.padding for img_info in imgs_info]  # type: ignore[union-attr]
        image_shape = imgs_info[0].img_shape  # type: ignore[union-attr, index]
        # Add additional conf threshold for saving maps with predicted classes,
        # since predictions can have less than 0.05 confidence
        conf_thr = explain_config.predicted_maps_conf_thr
        keep_ratio = imgs_info[0].keep_ratio  # type: ignore[union-attr, index]

        pred_labels = []
        for labels, scores in zip(predict_result_per_batch.labels, predict_result_per_batch.scores):  # type: ignore[union-attr, arg-type]
            if labels.shape == scores.shape:  # type: ignore[union-attr, attr-defined]
                # Filter out predictions with scores less than explain_config.predicted_maps_conf_thr
                pred_labels.append(labels[scores > conf_thr].tolist())  # type: ignore[operator]
            else:
                # Tv_* models case with a single predicted label as a scalar tensor with size zero
                labels_list = labels.tolist()  # type: ignore[attr-defined]
                labels_list = [labels_list] if isinstance(labels_list, int) else labels_list
                pred_labels.append(labels_list)

        processed_saliency_maps = process_saliency_maps(
            saliency_map,
            explain_config,
            pred_labels,
            ori_img_shapes,
            image_shape,
            paddings,
            keep_ratio,
        )
        predict_result_per_batch.saliency_map = processed_saliency_maps
        processed_predict_result.append(predict_result_per_batch)

    return processed_predict_result


def process_saliency_maps(
    saliency_map: list[np.ndarray],
    explain_config: ExplainConfig,
    pred_labels: list | None,
    ori_img_shapes: list,
    image_shape: tuple[int, int],
    paddings: list[tuple[int, int, int, int]],
    keep_ratio: bool,
) -> ProcessedSaliencyMaps:
    """Perform saliency map convertion to dict and post-processing."""
    if explain_config.target_explain_group == TargetExplainGroup.ALL:
        processed_saliency_maps = convert_maps_to_dict_all(saliency_map)
    elif explain_config.target_explain_group == TargetExplainGroup.PREDICTIONS:
        processed_saliency_maps = convert_maps_to_dict_predictions(saliency_map, pred_labels)
    elif explain_config.target_explain_group == TargetExplainGroup.IMAGE:
        processed_saliency_maps = convert_maps_to_dict_image(saliency_map)
    else:
        msg = f"Target explain group {explain_config.target_explain_group} is not supported."
        raise ValueError(msg)

    if keep_ratio:
        processed_saliency_maps = _crop_padded_map(processed_saliency_maps, image_shape, paddings)

    if explain_config.postprocess:
        for i in range(len(processed_saliency_maps)):
            processed_saliency_maps[i] = {
                key: postprocess(s_map, ori_img_shapes[i]) for key, s_map in processed_saliency_maps[i].items()
            }

    return processed_saliency_maps


def convert_maps_to_dict_all(saliency_map: list[np.ndarray]) -> list[dict[Any, np.array]]:
    """Convert salincy maps to dict for TargetExplainGroup.ALL."""
    processed_saliency_maps = []
    for maps_per_image in saliency_map:
        if maps_per_image.size == 0:
            processed_saliency_maps.append({0: np.zeros((1, 1, 1))})
            continue

        if maps_per_image.ndim != 3:
            msg = "Shape mismatch."
            raise ValueError(msg)

        explain_target_to_sal_map = dict(enumerate(maps_per_image))
        processed_saliency_maps.append(explain_target_to_sal_map)
    return processed_saliency_maps


def convert_maps_to_dict_predictions(
    saliency_map: list[np.ndarray],
    pred_labels: list | None,
) -> list[dict[Any, np.array]]:
    """Convert salincy maps to dict for TargetExplainGroup.PREDICTIONS."""
    if saliency_map[0].ndim != 3:
        msg = "Shape mismatch."
        raise ValueError(msg)
    if not pred_labels:
        return []

    processed_saliency_maps = []
    for i, maps_per_image in enumerate(saliency_map):
        explain_target_to_sal_map = {label: maps_per_image[label] for label in pred_labels[i] if pred_labels[i]}
        processed_saliency_maps.append(explain_target_to_sal_map)
    return processed_saliency_maps


def convert_maps_to_dict_image(saliency_map: list[np.ndarray]) -> list[dict[Any, np.array]]:
    """Convert salincy maps to dict for TargetExplainGroup.IMAGE."""
    if saliency_map[0].ndim != 2:
        msg = "Shape mismatch."
        raise ValueError(msg)
    return [{"map_per_image": map_per_image} for map_per_image in saliency_map]


def postprocess(saliency_map: np.ndarray, output_size: tuple[int, int] | None) -> np.ndarray:
    """Postprocess single saliency map."""
    if saliency_map.ndim != 2:
        msg = "Shape mismatch."
        raise ValueError(msg)

    if output_size:
        h, w = output_size
        saliency_map = cv2.resize(saliency_map, (w, h))
    return cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)


def _crop_padded_map(
    batch_saliency_map: list[dict[Any, np.array]],
    image_shape: tuple[int, int],
    paddings: list[tuple[int, int, int, int]],
) -> list[dict[Any, np.array]]:
    """Crop padded map."""
    # Padding: number of pixels to pad all borders (left, top, right, bottom)
    height, width = image_shape

    for i, image_map in enumerate(batch_saliency_map):
        left, top, right, bottom = paddings[i]
        left_ratio, top_ratio, right_ratio, bottom_ratio = left / width, top / height, right / width, bottom / height
        for class_idx, class_map in image_map.items():
            map_h, map_w = class_map.shape
            d_top, d_bottom = int(top_ratio * map_h), int(bottom_ratio * map_h)
            d_left, d_right = int(left_ratio * map_w), int(right_ratio * map_w)

            cropped_map = class_map[d_top : map_h - d_bottom, d_left : map_w - d_right]
            batch_saliency_map[i][class_idx] = cropped_map
    return batch_saliency_map
