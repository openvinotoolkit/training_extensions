# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom OTX Losses for Object Detection."""
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import CrossEntropyLoss
from .cross_focal_loss import CrossSigmoidFocalLoss, OrdinaryFocalLoss
from .smooth_l1_loss import L1Loss

__all__ = [
    "CrossEntropyLoss",
    "CrossSigmoidFocalLoss",
    "OrdinaryFocalLoss",
    "accuracy",
    "Accuracy",
    "L1Loss",
]
