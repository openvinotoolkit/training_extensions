# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module defines VisionDatasets."""

from .classification import MulticlassClsDataset, MultilabelClsDataset
from .detection import DetectionDataset
from .instance_segmentation import InstanceSegDataset
from .keypoint_detection import KeypointDetectionDataset
from .segmentation import SegmentationDataset
from .tile import TileDatasetFactory

__all__ = [
    "DetectionDataset",
    "InstanceSegDataset",
    "KeypointDetectionDataset",
    "MulticlassClsDataset",
    "MultilabelClsDataset",
    "SegmentationDataset",
    "TileDatasetFactory",
]
