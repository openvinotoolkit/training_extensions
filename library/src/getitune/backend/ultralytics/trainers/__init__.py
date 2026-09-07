# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Ultralytics trainer subclasses for getitune data bridge."""

from .base import GetiTuneBaseTrainer
from .classification import ClassificationTrainer, MultiLabelClassificationTrainer
from .detection import DetectionTrainer
from .instance_segmentation import SegmentationTrainer
from .semantic_segmentation import SemanticSegmentationTrainer
from .yolo_detr import YoloDetrTrainer


__all__ = [
    "ClassificationTrainer",
    "DetectionTrainer",
    "GetiTuneBaseTrainer",
    "MultiLabelClassificationTrainer",
    "SegmentationTrainer",
    "SemanticSegmentationTrainer",
    "YoloDetrTrainer",
]

