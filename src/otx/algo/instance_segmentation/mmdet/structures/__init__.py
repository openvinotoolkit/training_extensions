# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet structures."""
from .det_data_sample import DetDataSample, OptSampleList, SampleList

__all__ = [
    "DetDataSample",
    "SampleList",
    "OptSampleList",
]
