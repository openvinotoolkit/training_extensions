"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_overlaps import bbox_overlaps
from .transforms import (
    bbox2roi,
    empty_box_as,
    get_box_wh,
    scale_boxes,
)

__all__ = [
    "bbox_overlaps",
    "bbox2roi",
    "empty_box_as",
    "get_box_wh",
    "scale_boxes",
]
