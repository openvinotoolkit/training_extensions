"""Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
from typing import List, Optional, Dict

from datumaro.components.annotation import Annotation as DatumaroAnnotation
from datumaro.components.annotation import AnnotationType

from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.model_template import TaskType
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class DetectionDatasetAdapter(BaseDatasetAdapter):
    """Detection adapter inherited from BaseDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for object detection, and instance segmentation tasks
    """

    def __init__(
        self,
        task_type: TaskType,
        train_data_roots: Optional[str] = None,
        val_data_roots: Optional[str] = None,
        test_data_roots: Optional[str] = None,
        unlabeled_data_roots: Optional[str] = None,
    ):
        super().__init__(task_type, train_data_roots, val_data_roots, test_data_roots, unlabeled_data_roots)
    
    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Detection."""
        # Prepare label information
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        dataset_items: List[DatasetItemEntity] = []
        used_labels: List[int] = []
        for subset, subset_data in self.dataset.items():
            for datumaro_item in subset_data.get_subset(subset.name.lower()):
                image = Image(file_path=datumaro_item.media.path)
                shapes = []
                for ann in datumaro_item.annotations:
                    if self.task_type is TaskType.INSTANCE_SEGMENTATION and ann.type == AnnotationType.polygon:
                        shape = self._get_polygon_entity(ann, image.width, image.height)
                    if self.task_type is TaskType.DETECTION and ann.type == AnnotationType.bbox:
                        shape = self._get_normalized_bbox_entity(ann, image.width, image.height)
                    else:
                        shape = None
                        ValueError(f"{ann.type} is not supported.")
                    
                    if shape is not None:
                        shapes.append(shape)
                        
                    if ann.label not in used_labels:
                        used_labels.append(ann.label)
                dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                dataset_items.append(dataset_item)

        self.remove_unused_label_entities(used_labels)
        return DatasetEntity(items=dataset_items)
