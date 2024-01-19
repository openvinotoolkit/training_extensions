# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for export method."""
from __future__ import annotations

from dataclasses import dataclass

from otx.core.types.export import OTXExportFormatType, OTXExportPrecisionType


@dataclass
class ExplainConfig:
    """DTO for explain configuration."""

    target_explain_group: TargetExplainGroup
    postprocess: bool
