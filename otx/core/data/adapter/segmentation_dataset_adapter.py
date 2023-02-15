"""Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member, too-many-nested-blocks
from typing import List

from datumaro.components.annotation import AnnotationType
from datumaro.plugins.transforms import MasksToPolygons

from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter
import numpy as np
import cv2
from otx.api.entities.id import ID

class SegmentationDatasetAdapter(BaseDatasetAdapter):
    """Segmentation adapter inherited from BaseDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for semantic segmentation task
    """

    def remove_background_label(self):
        remove_bg = False
        for i, entity in enumerate(self.label_entities):
            if entity.name == "background":
                breakpoint()
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

        self.ignored_label = None

        if self.data_type_candidates[0] == "voc":
            self.set_voc_labels()
        
        if self.data_type_candidates[0] == "common_semantic_segmentation":
            self.set_common_labels()
        breakpoint()

        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    shapes: List[Annotation] = []
                    for ann in datumaro_item.annotations:
                        if ann.type == AnnotationType.mask:
                            # TODO: consider case -> didn't include the background information
                            cv2.imwrite("./tmp/mask.jpg", np.array(ann.image, dtype=np.uint8)*255)
                            datumaro_polygons = MasksToPolygons.convert_mask(ann)
                            for d_polygon in datumaro_polygons:
                                if d_polygon.label != 0 and d_polygon.label != self.ignored_label:
                                    d_polygon.label -= 1
                                    shapes.append(self._get_polygon_entity(d_polygon, image.width, image.height))
                                    if d_polygon.label not in used_labels:
                                        used_labels.append(d_polygon.label)

                    dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                    dataset_items.append(dataset_item)

        self.remove_unused_label_entities(used_labels)

        breakpoint()
        return DatasetEntity(items=dataset_items)
