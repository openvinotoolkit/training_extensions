# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom OTX Losses for Object Detection."""

from .cross_entropy_loss import CrossEntropyLoss
from .iou_loss import IoULoss
from .l1_loss import L1Loss


__all__ = ["CrossSigmoidFocalLoss, OrdinaryFocalLoss", "CrossEntropyLoss", "IoULoss", "L1Loss"]
