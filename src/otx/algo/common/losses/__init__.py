# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom loss implementations."""

from .cross_entropy_loss import CrossEntropyLoss
from .cross_focal_loss import CrossSigmoidFocalLoss
from .gfocal_loss import QualityFocalLoss
from .iou_loss import GIoULoss
from .smooth_l1_loss import L1Loss, smooth_l1_loss

__all__ = [
    "CrossEntropyLoss",
    "CrossSigmoidFocalLoss",
    "QualityFocalLoss",
    "GIoULoss",
    "L1Loss",
    "smooth_l1_loss",
]
