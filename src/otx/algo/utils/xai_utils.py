# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from otx.algo.hooks.recording_forward_hook import BaseRecordingForwardHook, MaskRCNNRecordingForwardHook
from otx.core.config.explain import ExplainConfig

if TYPE_CHECKING:
    from pathlib import Path


def get_processed_saliency_maps(
    raw_saliency_maps: list,
    explain_config: ExplainConfig,
    predictions: list | None,
    explain_hook: BaseRecordingForwardHook,
    work_dir: Path | None,
) -> list:
    """Implement saliency map filtering and post-processing."""
    if predictions is not None and isinstance(explain_hook, MaskRCNNRecordingForwardHook):
        # TODO: It is a temporary workaround. This function will be removed after we # noqa: TD003, TD002
        # refactor XAI logics into `OTXModel.forward_explain()`.

        # Mask-RCNN case, receive saliency maps from predictions
        raw_saliency_maps = explain_hook.func(predictions)

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
