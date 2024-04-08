"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""
# Copyright (c) OpenMMLab. All rights reserved.
from .anchor_generator import AnchorGenerator
from .utils import anchor_inside_flags

__all__ = [
    "AnchorGenerator",
    "anchor_inside_flags",
]
