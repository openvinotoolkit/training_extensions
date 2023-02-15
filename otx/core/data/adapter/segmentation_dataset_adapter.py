"""Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member, too-many-nested-blocks
from typing import List, Optional

from datumaro.components.annotation import AnnotationType
from datumaro.plugins.transforms import MasksToPolygons

from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.model_template import TaskType
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class SegmentationDatasetAdapter(BaseDatasetAdapter):
    """Segmentation adapter inherited from BaseDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for semantic segmentation task
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
        self.ignored_label = None

    def remove_background_label(self):
        """Remove background label in label entity set."""
        remove_bg = False
        for i, entity in enumerate(self.label_entities):
            if entity.name == "background":
                del self.label_entities[i]
                remove_bg = True
                break

        if remove_bg is True:
            for i, entity in enumerate(self.label_entities):
                entity.id = ID(i)

    def set_voc_labels(self):
        """Set labels for common_semantic_segmentation dataset."""
        # Remove background & ignored label in VOC from datumaro
        self.remove_background_label()
        for entity in self.label_entities:
            if entity.name == "ignored":
                self.ignored_label = int(entity.id)

    def set_common_labels(self):
        """Set labels for common_semantic_segmentation dataset."""
        # Remove background if in label_entities
        self.remove_background_label()

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Segmentation."""
        # Prepare label information
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        dataset_items: List[DatasetItemEntity] = []
        used_labels: List[int] = []
        if self.data_type_candidates[0] == "voc":
            self.set_voc_labels()

        if self.data_type_candidates[0] == "common_semantic_segmentation":
            self.set_common_labels()

        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    shapes: List[Annotation] = []
                    for ann in datumaro_item.annotations:
                        if ann.type == AnnotationType.mask:
                            # TODO: consider case -> didn't include the background information
                            datumaro_polygons = MasksToPolygons.convert_mask(ann)
                            for d_polygon in datumaro_polygons:
                                if d_polygon.label not in (0, self.ignored_label):
                                    d_polygon.label -= 1
                                    shapes.append(self._get_polygon_entity(d_polygon, image.width, image.height))
                                    if d_polygon.label not in used_labels:
                                        used_labels.append(d_polygon.label)

                    dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                    dataset_items.append(dataset_item)

        self.remove_unused_label_entities(used_labels)
        return DatasetEntity(items=dataset_items)
