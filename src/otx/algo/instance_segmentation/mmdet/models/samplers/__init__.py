"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""
# Copyright (c) OpenMMLab. All rights reserved.
from .random_sampler import RandomSampler
from .sampling_result import SamplingResult

__all__ = [
    "RandomSampler",
    "SamplingResult",
]
