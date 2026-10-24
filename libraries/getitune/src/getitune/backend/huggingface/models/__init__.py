# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face model wrappers, one per task."""

from .base import HFModel
from .classification import HFMulticlassClsModel, HFMultilabelClsModel
from .detection import HFDetectionModel
from .dinov3_classification import HFDinov3MulticlassClsModel, HFDinov3MultilabelClsModel
from .instance_segmentation import HFInstSegModel
from .semantic_segmentation import HFSemanticSegModel

__all__ = [
    "HFDetectionModel",
    "HFDinov3MulticlassClsModel",
    "HFDinov3MultilabelClsModel",
    "HFInstSegModel",
    "HFModel",
    "HFMulticlassClsModel",
    "HFMultilabelClsModel",
    "HFSemanticSegModel",
]
