# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet BBoxHead."""

from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead, Shared2FCBBoxHead

__all__ = [
    "BBoxHead",
    "ConvFCBBoxHead",
    "Shared2FCBBoxHead",
]
