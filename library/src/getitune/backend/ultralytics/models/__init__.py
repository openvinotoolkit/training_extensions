# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics models."""

from .base import UltralyticsModel
from .classification import UltralyticsMultiClassClsModel, UltralyticsMultiLabelClsModel
from .detection import UltralyticsDetectionModel
from .instance_segmentation import UltralyticsInstSegModel
from .semantic_segmentation import UltralyticsSemanticSegModel
from .yolo_detr import UltralyticsYoloDetrModel


__all__ = [
    "UltralyticsDetectionModel",
    "UltralyticsInstSegModel",
    "UltralyticsModel",
    "UltralyticsMultiClassClsModel",
    "UltralyticsMultiLabelClsModel",
    "UltralyticsSemanticSegModel",
    "UltralyticsYoloDetrModel",
]
