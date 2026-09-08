# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""multiclass classification models package."""

from .mobilenet_v3 import MobileNetV3MulticlassCls
from .timm_model import TimmModelMulticlassCls
from .vit import VisionTransformerMulticlassCls

__all__ = [
    "MobileNetV3MulticlassCls",
    "TimmModelMulticlassCls",
    "VisionTransformerMulticlassCls",
]
