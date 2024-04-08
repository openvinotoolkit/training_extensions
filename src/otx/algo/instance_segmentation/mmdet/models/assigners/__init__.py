"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from .assign_result import AssignResult
from .iou2d_calculator import BboxOverlaps2D
from .max_iou_assigner import MaxIoUAssigner

__all__ = [
    "AssignResult",
    "MaxIoUAssigner",
    "BboxOverlaps2D",
]
