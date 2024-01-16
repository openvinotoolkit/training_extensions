# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for export method."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from otx.core.types.export import OTXExportFormatType, OTXExportPrecisionType


@dataclass
class ExportConfig:
    """DTO for export configuration."""

    export_format: OTXExportFormatType
    precision: OTXExportPrecisionType

    input_height: int
    input_width: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize_mode: str
    pad_value: int
    swap_rgb: bool
    via_onnx: bool
    onnx_export_configuration: list[dict[str, Any]] | None = None
