# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""multiclass classification models package."""

from .efficientnet import EfficientNetMulticlassCls
from .mobilenet_v3 import MobileNetV3MulticlassCls
from .timm_model import TimmModelMulticlassCls
from .torchvision_model import TVModelMulticlassCls
from .vit import VisionTransformerMulticlassCls

__all__ = [
    "EfficientNetMulticlassCls",
    "MobileNetV3MulticlassCls",
    "TVModelMulticlassCls",
    "TimmModelMulticlassCls",
    "VisionTransformerMulticlassCls",
]
