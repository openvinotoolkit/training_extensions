# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone modules for OTX custom model."""

# from .otx_efficientnet import OTXEfficientNet
from .timm import TimmBackbone
from .mobilenet_v3 import OTXMobileNetV3
from .vision_transformer import OTXVisionTransformer

__all__ = ["OTXVisionTransformer", "TimmBackbone", "OTXMobileNetV3"]
