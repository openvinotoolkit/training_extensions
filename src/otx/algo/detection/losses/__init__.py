# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom OTX Losses for Object Detection."""

from .iou_loss import IoULoss
from .rtdetr_loss import DetrCriterion

__all__ = ["IoULoss", "DetrCriterion"]
