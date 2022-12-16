"""Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member, too-many-nested-blocks
from typing import Any, Tuple

from datumaro.components.annotation import Mask as DatumaroMask
from datumaro.plugins.transforms import MasksToPolygons

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
    NullAnnotationSceneEntity,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.core.data.base_dataset_adapter import BaseDatasetAdapter


class SegmentationDatasetAdapter(BaseDatasetAdapter):
    """Segmentation adapter inherited by BaseDatasetAdapter.
    It converts DatumaroDataset --> DatasetEntity for semantic segmentation task
    """

    def convert_to_otx_format(self, datumaro_dataset: dict) -> Tuple[DatasetEntity, LabelSchemaEntity]:
        """Convert DatumaroDataset to DatasetEntity for Segmentation."""
        # Prepare label information
        label_information = self._prepare_label_information(datumaro_dataset)
        label_entities = label_information["label_entities"]

        # Label schema
        label_schema = self._generate_default_label_schema(label_entities)

        dataset_items = []
        for subset, subset_data in datumaro_dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    shapes = []
                    for ann in datumaro_item.annotations:
                        if isinstance(ann, DatumaroMask):
                            if ann.label > 0:
                                datumaro_polygons = MasksToPolygons.convert_mask(ann)
                                for d_polygon in datumaro_polygons:
                                    shapes.append(
                                        Annotation(
                                            Polygon(
                                                points=[
                                                    Point(
                                                        x=d_polygon.points[i] / image.width,
                                                        y=d_polygon.points[i + 1] / image.height,
                                                    )
                                                    for i in range(0, len(d_polygon.points), 2)
                                                ]
                                            ),
                                            labels=[ScoredLabel(label=label_entities[d_polygon.label - 1])],
                                        )
                                    )
                    # Unlabeled dataset
                    annotation_scene = None  # type: Any
                    if len(shapes) == 0:
                        annotation_scene = NullAnnotationSceneEntity()
                    else:
                        annotation_scene = AnnotationSceneEntity(
                            kind=AnnotationSceneKind.ANNOTATION, annotations=shapes
                        )
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items), label_schema
