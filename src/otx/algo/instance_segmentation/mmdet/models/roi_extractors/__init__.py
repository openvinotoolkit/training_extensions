"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""
# Copyright (c) OpenMMLab. All rights reserved.
from .base_roi_extractor import BaseRoIExtractor
from .single_level_roi_extractor import SingleRoIExtractor

__all__ = ["BaseRoIExtractor", "SingleRoIExtractor"]
