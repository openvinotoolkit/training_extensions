# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2

from otx.algo.hooks.recording_forward_hook import BaseRecordingForwardHook, MaskRCNNRecordingForwardHook
from otx.core.config.explain import ExplainConfig
from otx.core.data.entity.base import OTXBatchPredEntityWithXAI
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntityWithXAI

if TYPE_CHECKING:
    from pathlib import Path


def get_processed_saliency_maps(
    explain_hook: BaseRecordingForwardHook,
    explain_config: ExplainConfig,
    predictions: list[Any] | list[OTXBatchPredEntityWithXAI | InstanceSegBatchPredEntityWithXAI] | None,
    work_dir: Path | None,
) -> list:
    """Implement saliency map filtering and post-processing."""
    # Optimize for memory <- TODO(negvet)
    raw_saliency_maps: list = []

    if predictions is not None:
        raw_saliency_maps = predictions[0].saliency_maps

    if predictions is not None and isinstance(explain_hook, MaskRCNNRecordingForwardHook):
        # TODO: It is a temporary workaround. This function will be removed after we # noqa: TD003, TD002
        # refactor XAI logics into `OTXModel.forward_explain()`.

        # Mask-RCNN case, receive saliency maps from predictions
        raw_saliency_maps = explain_hook.func(predictions)  # type: ignore[arg-type]

    if work_dir:
        # Temporary saving saliency map for image 0, class 0 (for tests)
        cv2.imwrite(str(work_dir / "saliency_map.tiff"), raw_saliency_maps[0][0])

    selected_saliency_maps = select_saliency_maps(raw_saliency_maps, explain_config, predictions)
    return process_saliency_maps(selected_saliency_maps, explain_config)


def select_saliency_maps(
    saliency_maps: list,
    explain_config: ExplainConfig,  # noqa: ARG001
    predictions: list | None,  # noqa: ARG001
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
