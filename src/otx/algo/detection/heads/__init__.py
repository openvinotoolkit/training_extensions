# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom head implementations for detection task."""

from .atss_head import ATSSHead
from .rtdetr_decoder import RTDETRTransformer
from .rtmdet_head import RTMDetSepBNHead
from .ssd_head import SSDHead
from .yolox_head import YOLOXHead

__all__ = ["ATSSHead", "RTMDetSepBNHead", "SSDHead", "YOLOXHead", "RTDETRTransformer"]
