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


class SegmentationDatasetAdapter(BaseDatasetAdapter):
    """Segmentation adapter inherited from BaseDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for semantic segmentation task
    """

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Segmentation."""
        # Prepare label information
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        dataset_items: List[DatasetItemEntity] = []
        used_labels: List[int] = []

        self.ignored_labels = None

        # voc datumaro does not remove ignored label.
        # so instead, when entity name is ignored, that polygon will be removed
        for entity in self.label_entities:
            if entity.name == "ignored":
                self.ignored_labels = int(entity.id)

        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    shapes: List[Annotation] = []
                    for ann in datumaro_item.annotations:
                        if ann.type == AnnotationType.mask:
                            # import numpy as np
                            # TODO: consider case -> didn't include the background information
                            # cv2.imwrite("./tmp/mask.jpg", np.array(ann.image, dtype=np.uint8)*255)
                            datumaro_polygons = MasksToPolygons.convert_mask(ann)
                            for d_polygon in datumaro_polygons:
                                if d_polygon.label != 0 and d_polygon.label != self.ignored_labels:
                                    shapes.append(self._get_polygon_entity(d_polygon, image.width, image.height))
                                    if d_polygon.label not in used_labels:
                                        used_labels.append(d_polygon.label)

                    dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                    dataset_items.append(dataset_item)

        self.remove_unused_label_entities(used_labels)
        breakpoint()

        return DatasetEntity(items=dataset_items)
