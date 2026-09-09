# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face model wrappers, one per task."""

from .base import HFModel
from .classification import HFMulticlassClsModel, HFMultilabelClsModel
from .detection import HFDetectionModel
from .instance_segmentation import HFInstSegModel
from .semantic_segmentation import HFSemanticSegModel

__all__ = [
    "HFDetectionModel",
    "HFInstSegModel",
    "HFModel",
    "HFMulticlassClsModel",
    "HFMultilabelClsModel",
    "HFSemanticSegModel",
]
