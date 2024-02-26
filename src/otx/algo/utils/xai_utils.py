# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2

from otx.core.config.explain import ExplainConfig
from otx.core.data.entity.base import OTXBatchPredEntityWithXAI
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntityWithXAI
from otx.core.types.explain import TargetExplainGroup

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def process_saliency_maps_in_pred_entity(
    predict_result: list[Any] | list[OTXBatchPredEntityWithXAI | InstanceSegBatchPredEntityWithXAI],
    explain_config: ExplainConfig,
    work_dir: Path | None = None,
) -> list[Any] | list[OTXBatchPredEntityWithXAI | InstanceSegBatchPredEntityWithXAI]:
    """Process saliency maps in PredEntity."""
    for batch_id in range(len(predict_result)):
        saliency_maps = predict_result[batch_id].saliency_maps
        pred_labels = predict_result[batch_id].labels  # type: ignore[union-attr]
        if pred_labels:
            pred_labels = [pred.tolist() for pred in pred_labels]

        processed_saliency_maps = process_saliency_maps(saliency_maps, explain_config, pred_labels)

        if processed_saliency_maps and work_dir:
            # Temporary saving random saliency map for image 0 (for tests)
            s_map_to_save = next(iter(processed_saliency_maps[0].values()))
            cv2.imwrite(str(work_dir / "saliency_map.tiff"), s_map_to_save)

        predict_result[batch_id].saliency_maps = processed_saliency_maps
    return predict_result


def process_saliency_maps(
    saliency_maps: list,
    explain_config: ExplainConfig,
    pred_labels: list | None,
) -> list[dict[Any, Any]]:
    """Perform saliency map convertion to dict and post-processing."""
    if explain_config.target_explain_group == TargetExplainGroup.ALL:
        processed_saliency_maps = convert_maps_to_dict_all(saliency_maps)
    elif explain_config.target_explain_group == TargetExplainGroup.PREDICTIONS:
        processed_saliency_maps = convert_maps_to_dict_predictions(saliency_maps, pred_labels)
    elif explain_config.target_explain_group == TargetExplainGroup.IMAGE:
        processed_saliency_maps = convert_maps_to_dict_predictions_image(saliency_maps)
    else:
        msg = f"Target explain group {explain_config.target_explain_group} is not supported."
        raise ValueError(msg)

    if explain_config.postprocess:
        for i in range(len(processed_saliency_maps)):
            processed_saliency_maps[i] = {key: postprocess(s_map) for key, s_map in processed_saliency_maps[i].items()}

    return processed_saliency_maps


def convert_maps_to_dict_all(saliency_maps: np.array) -> list[dict[Any, np.array]]:
    """Convert salincy maps to dict for TargetExplainGroup.ALL."""
    if saliency_maps[0].ndim != 3:
        raise ValueError

    processed_saliency_maps = []
    for maps_per_image in saliency_maps:
        explain_target_to_sal_map = dict(enumerate(maps_per_image))
        processed_saliency_maps.append(explain_target_to_sal_map)
    return processed_saliency_maps


def convert_maps_to_dict_predictions(saliency_maps: np.array, pred_labels: list | None) -> list[dict[Any, np.array]]:
    """Convert salincy maps to dict for TargetExplainGroup.PREDICTIONS."""
    if saliency_maps[0].ndim != 3:
        raise ValueError
    if not pred_labels:
        return []

    processed_saliency_maps = []
    for i, maps_per_image in enumerate(saliency_maps):
        explain_target_to_sal_map = {label: maps_per_image[label] for label in pred_labels[i] if pred_labels[i]}
        processed_saliency_maps.append(explain_target_to_sal_map)
    return processed_saliency_maps


def convert_maps_to_dict_predictions_image(saliency_maps: np.array) -> list[dict[Any, np.array]]:
    """Convert salincy maps to dict for TargetExplainGroup.IMAGE."""
    if saliency_maps[0].ndim != 2:
        raise ValueError
    return [{"map_per_image": map_per_image} for map_per_image in saliency_maps]


def postprocess(saliency_map: np.ndarray, output_size: tuple | list = (224, 224)) -> np.ndarray:
    """Postprocess single saliency map."""
    if saliency_map.ndim != 2:
        raise ValueError

    saliency_map = cv2.resize(saliency_map, output_size)
    return cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
