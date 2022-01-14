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

from ote_sdk.entities.model_template import TaskType


def get_dataset_class(task_type):
    """
    Returns a dataset class by task type.

    Args:
        task_type: A task type such as DETECTION, CLASSIFICATION, SEGMENTATION.
    """

    if task_type == TaskType.DETECTION:
        from .object_detection.dataset import ObjectDetectionDataset

        return ObjectDetectionDataset
    if task_type == TaskType.CLASSIFICATION:
        from .image_classification.dataset import ImageClassificationDataset

        return ImageClassificationDataset
    if task_type == TaskType.SEGMENTATION:
        from .semantic_segmentation.dataset import SemanticSegmentationDataset

        return SemanticSegmentationDataset
    if task_type == TaskType.ANOMALY_CLASSIFICATION:
        from .anomaly.dataset import AnomalyClassificationDataset

        return AnomalyClassificationDataset

    raise ValueError(f"Invalid task type: {task_type}")
