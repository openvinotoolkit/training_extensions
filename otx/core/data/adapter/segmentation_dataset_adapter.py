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
                                if d_polygon.label != 0:
                                    d_polygon.label -= 1
                                    shapes.append(self._get_polygon_entity(d_polygon, image.width, image.height))
                                    used_labels.append(d_polygon.label)

                    dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                    dataset_items.append(dataset_item)

        # Remove unused labels from label entities
        self.remove_unused_label_entities(used_labels)
        return DatasetEntity(items=dataset_items)
