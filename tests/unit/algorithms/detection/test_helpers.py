"""Collection of helper functions for unit tests of otx.algorithms.detection."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Any, Dict, List

import numpy as np

from otx.algorithms.detection.utils import generate_label_schema
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.model_template import TaskType, task_type_to_label_domain
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_helpers import generate_random_annotated_image

DEFAULT_DET_TEMPLATE_DIR = os.path.join("otx/algorithms/detection/configs/detection", "mobilenetv2_atss")
DEFAULT_ISEG_TEMPLATE_DIR = os.path.join(
    "otx/algorithms/detection/configs/instance_segmentation", "efficientnetb2b_maskrcnn"
)
DEFAULT_DET_RECIPE_CONFIG_PATH = "otx/recipes/stages/detection/incremental.py"
DEFAULT_ISEG_RECIPE_CONFIG_PATH = "otx/recipes/stages/instance-segmentation/incremental.py"


class MockImage(Image):
    """Mock class for Image entity."""

    @property
    def numpy(self) -> np.ndarray:
        """Returns empty numpy array"""

        return np.ndarray((256, 256))


class MockPipeline:
    """Mock class for data pipeline.

    It returns its inputs.
    """

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return results


def init_environment(params, model_template, task_type=TaskType.DETECTION):
    classes = ("rectangle", "ellipse", "triangle")
    label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=label_schema,
        model_template=model_template,
    )
    return environment


def generate_det_dataset(task_type, number_of_images=1):
    classes = ("rectangle", "ellipse", "triangle")
    label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))

    items = []
    for _ in range(number_of_images):
        image_numpy, annos = generate_random_annotated_image(
            image_width=640,
            image_height=480,
            labels=label_schema.get_labels(False),
        )
        # Convert shapes according to task
        for anno in annos:
            if task_type == TaskType.DETECTION:
                anno.shape = ShapeFactory.shape_as_rectangle(anno.shape)
            elif task_type == TaskType.INSTANCE_SEGMENTATION:
                anno.shape = ShapeFactory.shape_as_polygon(anno.shape)
        image = Image(data=image_numpy)
        annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=annos)
        items.append(DatasetItemEntity(media=image, annotation_scene=annotation_scene))
    dataset = DatasetEntity(items)
    return dataset, dataset.get_labels()


def generate_labels(length: int, domain: Domain) -> List[LabelEntity]:
    """Generate list of LabelEntity given length and domain."""

    output: List[LabelEntity] = []
    for i in range(length):
        output.append(LabelEntity(name=f"{i + 1}", domain=domain, id=ID(i + 1)))
    return output
