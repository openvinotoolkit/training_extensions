# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone modules for OTX custom model."""

from .efficientnet import OTXEfficientNet
from .mobilenet_v3 import OTXMobileNetV3
from .timm import TimmBackbone
from .torchvision import TorchvisionBackbone
from .vision_transformer import VisionTransformer

__all__ = ["OTXEfficientNet", "TimmBackbone", "OTXMobileNetV3", "VisionTransformer", "TorchvisionBackbone"]
