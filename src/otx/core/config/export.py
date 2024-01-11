# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for export method."""

from dataclasses import dataclass
from typing import Tuple

from otx.core.types.export import OTXExportPrecisionType, OTXExportFormatType


@dataclass
class ExportConfig:
    """DTO for export configuration."""

    format: OTXExportFormatType
    precision: OTXExportPrecisionType

    input_height: int
    input_width: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    resize_mode: str
    pad_value: int
    swap_rgb: bool