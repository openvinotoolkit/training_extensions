# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from otx.core.config.explain import ExplainConfig

if TYPE_CHECKING:
    from pathlib import Path


def get_processed_saliency_maps(
    raw_saliency_maps: list,
    explain_config: ExplainConfig | None,
    predictions: list | None,
    work_dir: Path | None,
) -> list:
    """Implement saliency map filtering and post-processing."""
    if work_dir:
        # Temporary saving saliency map for image 0, class 0 (for tests)
        cv2.imwrite(str(work_dir / "saliency_map.tiff"), raw_saliency_maps[0][0])

    selected_saliency_maps = select_saliency_maps(raw_saliency_maps, explain_config, predictions)
    return process_saliency_maps(selected_saliency_maps, explain_config)


def select_saliency_maps(
    saliency_maps: list,
    explain_config: ExplainConfig | None,  # noqa: ARG001
    predictions: list | None,  # noqa: ARG001
) -> list:
    """Select saliency maps in accordance with TargetExplainGroup."""
    # Implement <- TODO(negvet)
    return saliency_maps


def process_saliency_maps(
    saliency_maps: list,
    explain_config: ExplainConfig | None,  # noqa: ARG001
) -> list:
    """Postptocess saliency maps."""
    # Implement <- TODO(negvet)
    return saliency_maps
