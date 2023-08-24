"""Initial file for mmdetection assigners."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .custom_max_iou_assigner import CustomMaxIoUAssigner
from .dynamic_soft_label_assigner import DynamicSoftLabelAssigner

__all__ = ["CustomMaxIoUAssigner", "DynamicSoftLabelAssigner"]
