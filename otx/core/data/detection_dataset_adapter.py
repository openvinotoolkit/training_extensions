"""Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import Any, Tuple

# pylint: disable=invalid-name, too-many-locals, no-member
from datumaro.components.annotation import Bbox as DatumaroBbox
from datumaro.components.annotation import Polygon as DatumaroPolygon

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
from otx.api.entities.shapes.rectangle import Rectangle
from otx.core.data.base_dataset_adapter import BaseDatasetAdapter


class DetectionDatasetAdapter(BaseDatasetAdapter):
    """Detection adapter inherited by BaseDatasetAdapter.
    It converts DatumaroDataset --> DatasetEntity for object detection, and instance segmentation tasks
    """

    def convert_to_otx_format(self, datumaro_dataset: dict) -> Tuple[DatasetEntity, LabelSchemaEntity]:
        """Convert DatumaroDataset to DatasetEntity for Detection."""
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
                        if isinstance(ann, DatumaroPolygon):
                            shapes.append(
                                Annotation(
                                    Polygon(
                                        points=[
                                            Point(x=ann.points[i] / image.width, y=ann.points[i + 1] / image.height)
                                            for i in range(0, len(ann.points), 2)
                                        ]
                                    ),
                                    labels=[ScoredLabel(label=label_entities[ann.label])],
                                )
                            )
                            continue
                        if isinstance(ann, DatumaroBbox):
                            shapes.append(
                                Annotation(
                                    Rectangle(
                                        x1=ann.points[0] / image.width,
                                        y1=ann.points[1] / image.height,
                                        x2=ann.points[2] / image.width,
                                        y2=ann.points[3] / image.height,
                                    ),
                                    labels=[ScoredLabel(label=label_entities[ann.label])],
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
