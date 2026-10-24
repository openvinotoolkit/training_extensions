# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""multilabel classification models package."""

from .efficientnet import EfficientNetMultilabelCls
from .mobilenet_v3 import MobileNetV3MultilabelCls
from .timm_model import TimmModelMultilabelCls
from .torchvision_model import TVModelMultilabelCls
from .vit import VisionTransformerMultilabelCls

__all__ = [
    "EfficientNetMultilabelCls",
    "MobileNetV3MultilabelCls",
    "TVModelMultilabelCls",
    "TimmModelMultilabelCls",
    "VisionTransformerMultilabelCls",
]
