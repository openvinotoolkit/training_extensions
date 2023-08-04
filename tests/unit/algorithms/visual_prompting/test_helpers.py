""""Collection of helper functions for unit tests of otx.algorithms.visual_prompting."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import List, Optional, Tuple, Dict

import numpy as np

from otx.api.configuration.helper import create
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.color import Color
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from tests.test_helpers import generate_random_annotated_image

DEFAULT_VISUAL_PROMPTING_TEMPLATE_DIR = os.path.join("src/otx/algorithms/visual_prompting/configs", "sam_vit_b")

labels_names = ("rectangle", "ellipse", "triangle")


def generate_otx_label_schema(labels_names: List[str] = labels_names):
    label_domain = Domain.VISUAL_PROMPTING
    rgb = [int(i) for i in np.random.randint(0, 256, 3)]
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


def init_environment(model: Optional[ModelEntity] = None):
    model_template = parse_model_template(
        os.path.join(DEFAULT_VISUAL_PROMPTING_TEMPLATE_DIR, "template_experimental.yaml")
    )
    hyper_parameters = create(model_template.hyper_parameters.data)
    labels_schema = generate_otx_label_schema()
    environment = TaskEnvironment(
        model=model,
        hyper_parameters=hyper_parameters,
        label_schema=labels_schema,
        model_template=model_template,
    )
    return environment


class MockDatasetConfig:
    class _normalize:
        mean = [1.0, 1.0, 1.0]
        std = [0.0, 0.0, 0.0]

    def __init__(self, use_mask: bool = False):
        self.image_size: Tuple[int] = (4, 4)
        self.use_mask: bool = use_mask
        self.num_workers: int = 1
        self.train_batch_size: int = 1
        self.val_batch_size: int = 1
        self.test_batch_size: int = 1
        self.offset_bbox: int = 0
        self.normalize = self._normalize


class MockConfig:
    def __init__(self, use_mask: bool = False):
        self.dataset = MockDatasetConfig(use_mask=use_mask)
