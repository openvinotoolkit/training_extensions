# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet mask structures."""
from .mask_target import mask_target
from .structures import BitmapMasks, PolygonMasks, polygon_to_bitmap

__all__ = [
    "mask_target",
    "BitmapMasks",
    "PolygonMasks",
    "polygon_to_bitmap",
]
