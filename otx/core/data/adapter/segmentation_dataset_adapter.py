"""Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member, too-many-nested-blocks
from typing import Dict, List, Optional

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
        self.updated_label_id: Dict[int, int] = {}

    def remove_labels(self, label_names: List):
        """Remove background label in label entity set."""
        is_removed = False
        new_label_entities = []
        for i, entity in enumerate(self.label_entities):
            if entity.name not in label_names:
                new_label_entities.append(entity)
            else:
                is_removed = True

        for i, entity in enumerate(self.label_entities):
            self.updated_label_id[int(entity.id)] = i
            entity.id = ID(i)

        return is_removed

    def set_voc_labels(self):
        """Set labels for common_semantic_segmentation dataset."""
        # Remove background & ignored label in VOC from datumaro
        self.remove_labels(["background", "ignored"])

    def set_common_labels(self):
        """Set labels for common_semantic_segmentation dataset."""
        # Remove background if in label_entities
        is_removed = self.remove_labels(["background"])

        if is_removed is False:
            self.updated_label_id = {k + 1: v for k, v in self.updated_label_id.items()}

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
                                new_label = self.updated_label_id.get(d_polygon.label, None)
                                if new_label is not None:
                                    d_polygon.label = new_label
                                else:
                                    continue

                                shapes.append(self._get_polygon_entity(d_polygon, image.width, image.height))
                                if d_polygon.label not in used_labels:
                                    used_labels.append(d_polygon.label)

                    dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                    dataset_items.append(dataset_item)

        self.remove_unused_label_entities(used_labels)
        return DatasetEntity(items=dataset_items)
