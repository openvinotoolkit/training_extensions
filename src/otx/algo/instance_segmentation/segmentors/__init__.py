# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX instance segmentation segmentors."""

from .maskrcnn_tv import MaskRCNN
from .two_stage import TwoStageDetector

__all__ = ["MaskRCNN", "TwoStageDetector"]
