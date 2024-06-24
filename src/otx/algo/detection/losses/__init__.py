# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom OTX Losses for Object Detection."""

from .accuracy import accuracy
from .cross_entropy_loss import CrossEntropyLoss
from .cross_focal_loss import CrossSigmoidFocalLoss
from .dice_loss import DiceLoss
from .gfocal_loss import QualityFocalLoss
from .iou_loss import GIoULoss, IoULoss
from .smooth_l1_loss import L1Loss

__all__ = [
    "accuracy",
    "CrossEntropyLoss",
    "CrossSigmoidFocalLoss",
    "DiceLoss",
    "QualityFocalLoss",
    "GIoULoss",
    "IoULoss",
    "L1Loss",
]
