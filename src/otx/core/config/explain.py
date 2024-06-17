# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for export method."""
from __future__ import annotations

from dataclasses import dataclass

from otx.core.types.explain import TargetExplainGroup


@dataclass
class ExplainConfig:
    """Data Transfer Object (DTO) for explain configuration."""

    target_explain_group: TargetExplainGroup = TargetExplainGroup.ALL
    postprocess: bool = False
    crop_padded_map: bool = False
    predicted_maps_conf_thr: float = 0.3
