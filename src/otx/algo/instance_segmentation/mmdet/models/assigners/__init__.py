# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet Assigners."""
from .assign_result import AssignResult
from .iou2d_calculator import BboxOverlaps2D
from .max_iou_assigner import MaxIoUAssigner

__all__ = [
    "AssignResult",
    "MaxIoUAssigner",
    "BboxOverlaps2D",
]
