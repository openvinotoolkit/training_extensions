"""Initial file for mmdetection heads."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .cross_dataset_detector_head import CrossDatasetDetectorHead
from .custom_atss_head import CustomATSSHead

__all__ = [
    "CrossDatasetDetectorHead",
    "CustomATSSHead",
]
