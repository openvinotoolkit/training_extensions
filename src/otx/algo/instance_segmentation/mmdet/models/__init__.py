# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

""""MMDet model files."""
from .assigners import AssignResult, BboxOverlaps2D, MaxIoUAssigner
from .backbones import ResNet
from .coders import DeltaXYWHBBoxCoder
from .dense_heads import RPNHead
from .detectors import MaskRCNN
from .prior_generators import AnchorGenerator

__all__ = [
    "AssignResult",
    "MaxIoUAssigner",
    "BboxOverlaps2D",
    "ResNet",
    "DeltaXYWHBBoxCoder",
    "RPNHead",
    "MaskRCNN",
    "AnchorGenerator",
]
