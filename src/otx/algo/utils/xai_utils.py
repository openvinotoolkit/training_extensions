# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

from __future__ import annotations

from otx.core.config.explain import ExplainConfig


def get_processed_saliency_maps(
    raw_saliency_maps: list,
    explain_config: ExplainConfig | None,  # noqa: ARG001
    predictions: list | None,  # noqa: ARG001
) -> list | None:
    """Implement saliency map filtering and post-processing."""
    # Implement <- TODO(negvet)
    return raw_saliency_maps
