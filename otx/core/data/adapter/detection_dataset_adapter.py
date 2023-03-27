"""Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member, too-many-nested-blocks
from typing import List

from datumaro.components.annotation import AnnotationType

from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class DetectionDatasetAdapter(BaseDatasetAdapter):
    """Detection adapter inherited from BaseDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for object detection, and instance segmentation tasks
    """

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Detection."""
        # Prepare label information
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        dataset_items: List[DatasetItemEntity] = []
        used_labels: List[int] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    shapes = []
                    for ann in datumaro_item.annotations:
                        if (
                            self.task_type in (TaskType.INSTANCE_SEGMENTATION, TaskType.ROTATED_DETECTION)
                            and ann.type == AnnotationType.polygon
                        ):
                            if self._is_normal_polygon(ann):
                                shapes.append(self._get_polygon_entity(ann, image.width, image.height))
                        if self.task_type is TaskType.DETECTION and ann.type == AnnotationType.bbox:
                            if self._is_normal_bbox(ann.points[0], ann.points[1], ann.points[2], ann.points[3]):
                                shapes.append(self._get_normalized_bbox_entity(ann, image.width, image.height))

                        if ann.label not in used_labels:
                            used_labels.append(ann.label)

                    if (
                        len(shapes) > 0
                        or subset == Subset.UNLABELED
                        or (subset != Subset.TRAINING and len(datumaro_item.annotations) == 0)
                    ):
                        dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                        dataset_items.append(dataset_item)
        self.remove_unused_label_entities(used_labels)
        return DatasetEntity(items=dataset_items)
