"""File system based datasets registry."""

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

from otx.api.entities.model_template import TaskType


# pylint: disable=too-many-return-statements
def get_dataset_class(task_type):
    """Returns a dataset class by task type.

    Args:
        task_type: A task type such as ANOMALY_CLASSIFICATION, ANOMALY_DETECTION, ANOMALY_SEGMENTATION,
        CLASSIFICATION, INSTANCE_SEGMENTATION, DETECTION, CLASSIFICATION, ROTATED_DETECTION, SEGMENTATION.
    """

    if task_type == TaskType.ANOMALY_CLASSIFICATION:
        from otx.algorithms.anomaly.adapters.anomalib.data.dataset import (
            AnomalyClassificationDataset,
        )

        return AnomalyClassificationDataset
    if task_type == TaskType.ANOMALY_DETECTION:
        from otx.algorithms.anomaly.adapters.anomalib.data.dataset import (
            AnomalyDetectionDataset,
        )

        return AnomalyDetectionDataset
    if task_type == TaskType.ANOMALY_SEGMENTATION:
        from otx.algorithms.anomaly.adapters.anomalib.data.dataset import (
            AnomalySegmentationDataset,
        )

        return AnomalySegmentationDataset
    if task_type == TaskType.CLASSIFICATION:
        from .image_classification.dataset import ImageClassificationDataset

        return ImageClassificationDataset
    if task_type == TaskType.DETECTION:
        from .object_detection.dataset import ObjectDetectionDataset

        return ObjectDetectionDataset
    if task_type == TaskType.INSTANCE_SEGMENTATION:
        from .instance_segmentation.dataset import InstanceSegmentationDataset

        return InstanceSegmentationDataset
    if task_type == TaskType.ROTATED_DETECTION:
        from .rotated_detection.dataset import RotatedDetectionDataset

        return RotatedDetectionDataset
    if task_type == TaskType.SEGMENTATION:
        from .semantic_segmentation.dataset import SemanticSegmentationDataset

        return SemanticSegmentationDataset
    if task_type == TaskType.ACTION_CLASSIFICATION:
        from .action_classification.dataset import ActionClassificationDataset

        return ActionClassificationDataset
    if task_type == TaskType.ACTION_DETECTION:
        from .action_detection.dataset import ActionDetectionDataset

        return ActionDetectionDataset

    raise ValueError(f"Invalid task type: {task_type}")
