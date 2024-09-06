# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom OTX Losses for Object Detection."""

from .atss_loss import ATSSCriterion
from .rtdetr_loss import DetrCriterion
from .rtmdet_loss import RTMDetCriterion
from .ssd_loss import SSDCriterion
from .yolo_loss import YOLOCriterion
from .yolox_loss import YOLOXCriterion

__all__ = [
    "ATSSCriterion",
    "DetrCriterion",
    "RTMDetCriterion",
    "SSDCriterion",
    "YOLOCriterion",
    "YOLOXCriterion",
]
