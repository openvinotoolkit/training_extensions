# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom head implementations for detection task."""

from .custom_anchor_generator import SSDAnchorGeneratorClustered
from .custom_atss_head import CustomATSSHead
from .custom_ssd_head import SSDHead
from .yolox_head import YOLOXHead

__all__ = ["SSDAnchorGeneratorClustered", "CustomATSSHead", "SSDHead", "YOLOXHead"]
