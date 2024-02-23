# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2

from otx.core.config.explain import ExplainConfig
from otx.core.data.entity.base import OTXBatchPredEntityWithXAI
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntityWithXAI

if TYPE_CHECKING:
    from pathlib import Path


def get_processed_saliency_maps(
    predictions: list[Any] | list[OTXBatchPredEntityWithXAI | InstanceSegBatchPredEntityWithXAI] | None,
    explain_config: ExplainConfig,
    work_dir: Path | None,
) -> list:
    """Implement saliency map filtering and post-processing."""
    # Optimize for memory <- TODO(negvet)
    if predictions is None:
        msg = "Model predictions are not available."
        raise ValueError(msg)

    raw_saliency_maps = predictions[0].saliency_maps
    pred_labels = predictions[0].labels

    if raw_saliency_maps and work_dir:
        # Temporary saving saliency map for image 0, class 0 (for tests)
        cv2.imwrite(str(work_dir / "saliency_map.tiff"), raw_saliency_maps[0][0])

    selected_saliency_maps = select_saliency_maps(raw_saliency_maps, explain_config, pred_labels)
    return process_saliency_maps(selected_saliency_maps, explain_config)


def select_saliency_maps(
    saliency_maps: list,
    explain_config: ExplainConfig,  # noqa: ARG001
    pred_labels: list | None,  # noqa: ARG001
) -> list:
    """Select saliency maps in accordance with TargetExplainGroup."""
    # Implement <- TODO(negvet)
    return saliency_maps


def process_saliency_maps(
    saliency_maps: list,
    explain_config: ExplainConfig,  # noqa: ARG001
) -> list:
    """Postptocess saliency maps."""
    # Implement <- TODO(negvet)
    return saliency_maps


def add_saliency_maps_to_dataset_item(
    dataset_item: DatasetItemEntity,
    saliency_map: Union[List[Optional[np.ndarray]], np.ndarray],
    model: Optional[ModelEntity],
    labels: List[LabelEntity],
    predicted_scored_labels: Optional[List[ScoredLabel]] = None,
    explain_predicted_classes: bool = True,
    process_saliency_maps: bool = False,
):
    """Add saliency maps (2D array for class-agnostic saliency map,
    3D array or list or 2D arrays for class-wise saliency maps) to a single dataset item."""
    if isinstance(saliency_map, list):
        class_wise_saliency_map = True
    elif isinstance(saliency_map, np.ndarray):
        if saliency_map.ndim == 2:
            class_wise_saliency_map = False
        elif saliency_map.ndim == 3:
            class_wise_saliency_map = True
        else:
            raise ValueError(f"Saliency map has to be 2 or 3-dimensional array, " f"but got {saliency_map.ndim} dims.")
    else:
        raise TypeError("Check saliency_map, it has to be list or np.ndarray.")

    if class_wise_saliency_map:
        # Multiple saliency maps per image (class-wise saliency map), support e.g. ReciproCAM
        if explain_predicted_classes:
            # Explain only predicted classes
            if predicted_scored_labels is None:
                raise ValueError("To explain only predictions, list of predicted scored labels have to be provided.")

            explain_targets = set()
            for scored_label in predicted_scored_labels:
                if scored_label.label is not None:  # Check for an empty label
                    explain_targets.add(scored_label.label)
        else:
            # Explain all classes
            explain_targets = set(labels)

        for class_id, class_wise_saliency_map in enumerate(saliency_map):
            label = labels[class_id]
            if class_wise_saliency_map is not None and label in explain_targets:
                if process_saliency_maps:
                    class_wise_saliency_map = get_actmap(
                        class_wise_saliency_map, (dataset_item.width, dataset_item.height)
                    )
                saliency_media = ResultMediaEntity(
                    name=label.name,
                    type="saliency_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=class_wise_saliency_map,
                    roi=dataset_item.roi,
                    label=label,
                )
                dataset_item.append_metadata_item(saliency_media, model=model)
    else:
        # Single saliency map per image, support e.g. ActivationMap
        if process_saliency_maps:
            saliency_map = get_actmap(saliency_map, (dataset_item.width, dataset_item.height))
        saliency_media = ResultMediaEntity(
            name="Saliency Map",
            type="saliency_map",
            annotation_scene=dataset_item.annotation_scene,
            numpy=saliency_map,
            roi=dataset_item.roi,
        )
        dataset_item.append_metadata_item(saliency_media, model=model)


def get_actmap(
    saliency_map: np.ndarray,
    output_res: Union[tuple, list],
) -> np.ndarray:
    """Get activation map (heatmap)  from saliency map.

    It will return activation map from saliency map

    Args:
        saliency_map (np.ndarray): Saliency map with pixel values from 0-255
        output_res (Union[tuple, list]): Output resolution

    Returns:
        saliency_map (np.ndarray): [H, W, 3] colormap, more red means more salient

    """
    if len(saliency_map.shape) == 3:
        saliency_map = saliency_map[0]

    saliency_map = cv2.resize(saliency_map, output_res)
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    return saliency_map


def save_saliency_output(
    process_saliency_maps: bool,
    img: np.array,
    saliency_map: np.array,
    save_dir: str,
    fname: str,
    weight: float = 0.3,
) -> None:
    """Saves processed saliency map (with image overlay) or raw saliency map."""
    if process_saliency_maps:
        # Saves processed saliency map
        overlay = img * weight + saliency_map * (1 - weight)
        overlay[overlay > 255] = 255
        overlay = overlay.astype(np.uint8)

        cv2.imwrite(f"{osp.join(save_dir, fname)}_saliency_map.png", saliency_map)
        cv2.imwrite(f"{osp.join(save_dir, fname)}_overlay_img.png", overlay)
    else:
        # Saves raw, low-resolution saliency map
        cv2.imwrite(f"{osp.join(save_dir, fname)}_saliency_map.tiff", saliency_map)
