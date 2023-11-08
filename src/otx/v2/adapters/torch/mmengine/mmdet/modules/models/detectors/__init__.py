"""Initial file for mmdetection detectors."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .custom_atss_detector import CustomATSS
from .custom_maskrcnn_detector import CustomMaskRCNN

__all__ = ["CustomATSS", "CustomMaskRCNN"]
