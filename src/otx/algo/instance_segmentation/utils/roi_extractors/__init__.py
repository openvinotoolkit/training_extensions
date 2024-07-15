# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""ROI extractors for instance segmentation task."""

from .roi_align import OTXRoIAlign
from .single_level_roi_extractor import SingleRoIExtractor

__all__ = ["OTXRoIAlign", "SingleRoIExtractor"]
