# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom head implementations for detection task."""

from .atss_head import ATSSHead
from .dfine_decoder import DFINETransformer
from .rtdetr_decoder import RTDETRTransformer
from .ssd_head import SSDHead
from .yolox_head import YOLOXHead

__all__ = [
    "ATSSHead",
    "DFINETransformer",
    "RTDETRTransformer",
    "SSDHead",
    "YOLOXHead",
]
