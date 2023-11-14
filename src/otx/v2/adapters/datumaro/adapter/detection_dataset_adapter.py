"""Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List

from datumaro.components.annotation import AnnotationType as DatumAnnotationType

from otx.v2.api.entities.dataset_item import DatasetItemEntityWithID
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType

from .datumaro_dataset_adapter import DatumaroDatasetAdapter


class DetectionDatasetAdapter(DatumaroDatasetAdapter):
    """Detection adapter inherited from DatumaroDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for object detection, and instance segmentation tasks
    """

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Detection."""
        if not hasattr(self, "label_entities"):
            super().get_label_schema()
        dataset_items: List[DatasetItemEntityWithID] = []
        used_labels: List[int] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = self.datum_media_2_otx_media(datumaro_item.media)
                    shapes = []
                    for ann in datumaro_item.annotations:
                        if (
                            self.task_type in (TaskType.INSTANCE_SEGMENTATION, TaskType.ROTATED_DETECTION)
                            and ann.type == DatumAnnotationType.polygon
                        ) and self._is_normal_polygon(ann):
                            shapes.append(self._get_polygon_entity(ann, image.width, image.height))
                        if self.task_type is TaskType.DETECTION and ann.type == DatumAnnotationType.bbox:
                            if self._is_normal_bbox(ann.points[0], ann.points[1], ann.points[2], ann.points[3]):
                                shapes.append(self._get_normalized_bbox_entity(ann, image.width, image.height))

                        if ann.label not in used_labels:
                            used_labels.append(ann.label)

                    if (
                        len(shapes) > 0
                        or subset == Subset.UNLABELED
                        or (subset != Subset.TRAINING and len(datumaro_item.annotations) == 0)
                    ):
                        dataset_item = DatasetItemEntityWithID(
                            image,
                            self._get_ann_scene_entity(shapes),
                            subset=subset,
                            id_=datumaro_item.id,
                        )
                        dataset_items.append(dataset_item)
        
        self.remove_unused_label_entities(used_labels)
        return DatasetEntity(items=dataset_items)
