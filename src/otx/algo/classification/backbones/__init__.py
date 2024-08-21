# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone modules for OTX custom model."""

from .efficientnet import EfficientNetBackbone
from .mobilenet_v3 import MobileNetV3Backbone
from .timm import TimmBackbone
from .torchvision import TorchvisionBackbone
from .vision_transformer import VisionTransformer

__all__ = ["EfficientNetBackbone", "TimmBackbone", "MobileNetV3Backbone", "VisionTransformer", "TorchvisionBackbone"]
