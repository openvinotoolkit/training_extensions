""""Collection of helper functions for unit tests of visual_prompting."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import ClassVar

import numpy as np
from otx.v2.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.v2.api.entities.color import Color
from otx.v2.api.entities.dataset_item import DatasetItemEntity
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.image import Image
from otx.v2.api.entities.label import Domain, LabelEntity
from otx.v2.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.v2.api.entities.shapes.ellipse import Ellipse
from otx.v2.api.entities.shapes.polygon import Point, Polygon
from otx.v2.api.entities.shapes.rectangle import Rectangle
from otx.v2.api.entities.subset import Subset

from tests.v2.test_helpers import generate_random_annotated_image

labels_names = ["rectangle", "ellipse", "triangle"]



def generate_otx_label_schema(labels_names: list[str] = labels_names) -> LabelSchemaEntity:
    label_domain = Domain.VISUAL_PROMPTING
    rng = np.random.default_rng()
    rgb = [int(i) for i in rng.integers(0, 256, 3)]
    colors = [Color(*rgb) for _ in range(len(labels_names))]
    not_empty_labels = [
        LabelEntity(name=name, color=colors[i], domain=label_domain, id=i) for i, name in enumerate(labels_names)
    ]
    empty_label = LabelEntity(
        name="Empty label",
        color=Color(42, 43, 46),
        is_empty=True,
        domain=label_domain,
        id=len(not_empty_labels),
    )

    label_schema = LabelSchemaEntity()
    exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
    empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
    label_schema.add_group(exclusive_group)
    label_schema.add_group(empty_group)
    return label_schema


def generate_visual_prompting_dataset(use_mask: bool = False) -> DatasetEntity:
    items = []
    labels_schema = generate_otx_label_schema()
    labels_list = labels_schema.get_labels(False)
    for subset in [Subset.TRAINING, Subset.VALIDATION, Subset.TESTING, Subset.NONE]:
        image_numpy, shapes = generate_random_annotated_image(
            image_width=640,
            image_height=480,
            labels=labels_list,
            max_shapes=20,
            min_size=50,
            max_size=100,
            random_seed=None,
            use_mask_as_annotation=use_mask,
        )

        out_shapes = []
        for shape in shapes:
            shape_labels = shape.get_labels(include_empty=True)

            in_shape = shape.shape
            if use_mask:
                if isinstance(in_shape, Image):
                    out_shapes.append(shape)
            else:
                if isinstance(in_shape, Rectangle):
                    points = [
                        Point(in_shape.x1, in_shape.y1),
                        Point(in_shape.x2, in_shape.y1),
                        Point(in_shape.x2, in_shape.y2),
                        Point(in_shape.x1, in_shape.y2),
                    ]
                elif isinstance(in_shape, Ellipse):
                    points = [Point(x, y) for x, y in in_shape.get_evenly_distributed_ellipse_coordinates()]
                elif isinstance(in_shape, Polygon):
                    points = in_shape.points

                out_shapes.append(Annotation(Polygon(points=points), labels=shape_labels))
        image = Image(data=image_numpy)
        annotation = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=out_shapes)
        items.append(DatasetItemEntity(media=image, annotation_scene=annotation, subset=subset))

    return DatasetEntity(items)


class MockDatasetConfig:
    class Normalize:
        mean: ClassVar = [1.0, 1.0, 1.0]
        std: ClassVar = [0.0, 0.0, 0.0]

    def __init__(self, use_mask: bool = False) -> None:
        self.image_size: tuple[int, int] = (4, 4)
        self.use_mask: bool = use_mask
        self.num_workers: int = 1
        self.train_batch_size: int = 1
        self.val_batch_size: int = 1
        self.test_batch_size: int = 1
        self.offset_bbox: int = 0
        self.normalize = self.Normalize


class MockConfig:
    def __init__(self, use_mask: bool = False) -> None:
        self.dataset = MockDatasetConfig(use_mask=use_mask)
