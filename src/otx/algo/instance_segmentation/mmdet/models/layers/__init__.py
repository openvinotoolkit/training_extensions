"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""
# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_nms import multiclass_nms
from .res_layer import ResLayer

# yapf: enable

__all__ = [
    "multiclass_nms",
    "ResLayer",
]
