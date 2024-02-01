# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

from __future__ import annotations

from otx.core.config.explain import ExplainConfig


def get_processed_saliency_maps(
    raw_saliency_maps: list,
    explain_config: ExplainConfig | None,
    predictions: list | None,
) -> list:
    """Implement saliency map filtering and post-processing."""
    # Implement <- TODO(negvet)
    selected_saliency_maps = select_saliency_maps(raw_saliency_maps, explain_config, predictions)
    return process_saliency_maps(selected_saliency_maps, explain_config)


def select_saliency_maps(
    saliency_maps: list,
    explain_config: ExplainConfig | None,  # noqa: ARG001
    predictions: list | None,  # noqa: ARG001
) -> list:
    """Select saliency maps in accordance with TargetExplainGroup."""
    return saliency_maps


def process_saliency_maps(
    saliency_maps: list,
    explain_config: ExplainConfig | None,  # noqa: ARG001
) -> list:
    """Postptocess saliency maps."""
    return saliency_maps
