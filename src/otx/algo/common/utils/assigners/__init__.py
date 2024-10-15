# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom assigner implementations."""

from .dynamic_soft_label_assigner import DynamicSoftLabelAssigner
from .iou2d_calculator import BboxOverlaps2D
from .max_iou_assigner import MaxIoUAssigner

__all__ = ["DynamicSoftLabelAssigner", "BboxOverlaps2D", "MaxIoUAssigner"]
