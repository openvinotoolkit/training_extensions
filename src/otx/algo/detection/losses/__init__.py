# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom OTX Losses for Object Detection."""

from .atss_loss import ATSSCriterion
from .iou_loss import IoULoss
from .rtdetr_loss import DetrCriterion
from .rtmdet_loss import RTMDetCriterion
from .ssd_loss import SSDCriterion
from .yolox_loss import YOLOXCriterion

__all__ = ["ATSSCriterion", "IoULoss", "DetrCriterion", "RTMDetCriterion", "SSDCriterion", "YOLOXCriterion"]
