"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from .mask_target import mask_target
from .structures import BitmapMasks, PolygonMasks, polygon_to_bitmap

__all__ = [
    "mask_target",
    "BitmapMasks",
    "PolygonMasks",
    "polygon_to_bitmap",
]
