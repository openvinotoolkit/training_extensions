# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backbone modules for getitune custom model."""

from .efficientnet import EfficientNetBackbone
from .mobilenet_v3 import MobileNetV3Backbone
from .timm import TimmBackbone
from .torchvision import TorchvisionBackbone
from .vision_transformer import VisionTransformerBackbone

__all__ = [
    "EfficientNetBackbone",
    "MobileNetV3Backbone",
    "TimmBackbone",
    "TorchvisionBackbone",
    "VisionTransformerBackbone",
]
