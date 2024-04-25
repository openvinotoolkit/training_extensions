# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom OTX Losses for Object Detection."""
from .accuracy import accuracy
from .cross_entropy_loss import CrossEntropyLoss
from .cross_focal_loss import CrossSigmoidFocalLoss
from .smooth_l1_loss import L1Loss
from .iou_loss import IoULoss


__all__ = [
    "CrossEntropyLoss",
    "CrossSigmoidFocalLoss",
    "accuracy",
    "L1Loss",
    "IoULoss",
]
