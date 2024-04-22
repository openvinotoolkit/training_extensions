# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet Layers."""

from .bbox_nms import multiclass_nms
from .res_layer import ResLayer
from .transformer import PatchEmbed, PatchMerging

__all__ = [
    "multiclass_nms",
    "ResLayer",
    "PatchEmbed",
    "PatchMerging",
]
