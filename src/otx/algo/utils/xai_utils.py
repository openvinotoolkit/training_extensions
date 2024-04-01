# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

# TODO(gzalessk): Typings in this file is too weak or wrong. It should be fixed.
# For example, `pred_labels: list | None` has no object typing containered in the list.
# On the other hand, process_saliency_maps should not produce list of dictionaries
# (`list[dict[str, np.ndarray | torch.Tensor]]`).
# This is because the output will be assigned to OTXBatchPredEntity.saliency_map,
# but this has `list[np.ndarray | torch.Tensor]` typing, so that it makes a mismatch.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from datumaro import Image

from otx.core.config.explain import ExplainConfig
from otx.core.data.entity.classification import (
    HlabelClsBatchPredEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.data.entity.detection import DetBatchPredEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity
from otx.core.types.explain import TargetExplainGroup

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS

    from otx.core.data.module import OTXDataModule

ProcessedSaliencyMaps = list[dict[str, np.ndarray | torch.Tensor]]
OTXBatchPredEntitiesSupportXAI = (
    MulticlassClsBatchPredEntity
    | MultilabelClsBatchPredEntity
    | HlabelClsBatchPredEntity
    | DetBatchPredEntity
    | InstanceSegBatchPredEntity
)


def process_saliency_maps_in_pred_entity(
    predict_result: list[OTXBatchPredEntitiesSupportXAI],
    explain_config: ExplainConfig,
) -> list[OTXBatchPredEntitiesSupportXAI]:
    """Process saliency maps in PredEntity."""

    def _process(predict_result_per_batch: OTXBatchPredEntitiesSupportXAI) -> OTXBatchPredEntitiesSupportXAI:
        saliency_map: list[np.ndarray] = [
            saliency_map.cpu().numpy() if isinstance(saliency_map, torch.Tensor) else saliency_map
            for saliency_map in predict_result_per_batch.saliency_map
        ]
        imgs_info = predict_result_per_batch.imgs_info
        ori_img_shapes = [img_info.ori_shape for img_info in imgs_info]
        pred_labels = [pred.tolist() for pred in predict_result_per_batch.labels]

        processed_saliency_maps = process_saliency_maps(saliency_map, explain_config, pred_labels, ori_img_shapes)

        return predict_result_per_batch.wrap(saliency_map=processed_saliency_maps)

    return [_process(predict_result_per_batch) for predict_result_per_batch in predict_result]


def process_saliency_maps(
    saliency_map: list[np.ndarray],
    explain_config: ExplainConfig,
    pred_labels: list | None,
    ori_img_shapes: list,
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


def dump_saliency_maps(
    predict_result: list[OTXBatchPredEntitiesSupportXAI],
    explain_config: ExplainConfig,
    datamodule: EVAL_DATALOADERS | OTXDataModule,
    output_dir: Path,
    weight: float = 0.3,
) -> None:
    """Sumps saliency maps (raw and with overlay)."""
    output_dir = output_dir / "saliency_map"
    output_dir.mkdir(parents=True, exist_ok=True)

    for predict_result_per_batch in predict_result:
        saliency_map = predict_result_per_batch.saliency_map
        imgs_info = predict_result_per_batch.imgs_info
        for pred_index in range(len(saliency_map)):
            img_id = imgs_info[pred_index].img_idx
            img_data, image_save_name = _get_image_data_name(datamodule, img_id)

            for class_id, s_map in saliency_map[pred_index].items():
                file_name_map = Path(image_save_name + "_class_" + str(class_id) + "_saliency_map.png")
                save_path_map = output_dir / file_name_map
                cv2.imwrite(str(save_path_map), s_map)

                if explain_config.postprocess:
                    file_name_overlay = Path(image_save_name + "_class_" + str(class_id) + "_overlay.png")
                    save_path_overlay = output_dir / file_name_overlay
                    overlay = _get_overlay(img_data, s_map, weight)
                    cv2.imwrite(str(save_path_overlay), overlay)


def _get_image_data_name(
    datamodule: EVAL_DATALOADERS | OTXDataModule,
    img_id: int,
    subset_name: str = "test",
) -> tuple[np.array, str]:
    subset = datamodule.subsets[subset_name]
    image_name = subset.ids[img_id]
    item = subset.dm_subset.get(id=image_name, subset=subset_name)
    img = item.media_as(Image)
    img_data, _ = subset._get_img_data_and_shape(img)  # noqa: SLF001
    image_save_name = "".join([char if char.isalnum() else "_" for char in image_name])
    return img_data, image_save_name


def _get_overlay(img: np.ndarray, s_map: np.ndarray, weight: float = 0.3) -> np.ndarray:
    overlay = img * weight + s_map * (1 - weight)
    overlay[overlay > 255] = 255
    return overlay.astype(np.uint8)
