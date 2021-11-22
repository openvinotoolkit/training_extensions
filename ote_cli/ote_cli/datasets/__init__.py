"""
File system based datasets registry.
"""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Optional, Type

from ote_sdk.entities.datasets import DatasetEntity

try:
    ObjectDetectionDataset: Optional[Type[DatasetEntity]]
    from .object_detection.dataset import ObjectDetectionDataset
except ImportError:
    ObjectDetectionDataset = None

try:
    ImageClassificationDataset: Optional[Type[DatasetEntity]]
    from .image_classification.dataset import ImageClassificationDataset
except ImportError:
    ImageClassificationDataset = None

try:
    SemanticSegmentationDataset: Optional[Type[DatasetEntity]]
    from .semantic_segmentation.dataset import SemanticSegmentationDataset
except ImportError:
    SemanticSegmentationDataset = None

assert any(
    (ObjectDetectionDataset, ImageClassificationDataset, SemanticSegmentationDataset)
)


def get_dataset_class(task_type):
    """
    Returns a dataset class by task type.

    Args:
        task_type: A task type such as detection, classification, segmentation.
    """
    registry = {
        "detection": ObjectDetectionDataset,
        "classification": ImageClassificationDataset,
        "segmentation": SemanticSegmentationDataset,
    }

    return registry[str(task_type).lower()]


__all__ = [
    "ObjectDetectionDataset",
    "ImageClassificationDataset",
    "SemanticSegmentationDataset",
    "get_dataset_class",
]
