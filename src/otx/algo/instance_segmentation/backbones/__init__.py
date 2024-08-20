# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom backbone implementations for instance segmentation task."""

from .backbone_constructor import MaskRCNNBackbone
from .swin import SwinTransformer

__all__ = ["SwinTransformer", "MaskRCNNBackbone"]
