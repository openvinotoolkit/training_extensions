"""Collection of helper functions for unit tests of otx.algorithms.detection."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
from typing import Any, Dict, List

import numpy as np

from otx.algorithms.detection.utils import generate_label_schema
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.dataset_item import DatasetItemEntityWithID
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.model_template import TaskType, task_type_to_label_domain
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_helpers import generate_random_annotated_image

DEFAULT_DET_MODEL_CONFIG_PATH = "src/otx/algorithms/detection/configs/detection/mobilenetv2_atss/model.py"
DEFAULT_ISEG_MODEL_CONFIG_PATH = (
    "src/otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/model.py"
)

DEFAULT_DET_TEMPLATE_DIR = os.path.join("src/otx/algorithms/detection/configs/detection", "mobilenetv2_atss")
DEFAULT_ISEG_TEMPLATE_DIR = os.path.join(
    "src/otx/algorithms/detection/configs/instance_segmentation", "efficientnetb2b_maskrcnn"
)
DEFAULT_DET_RECIPE_CONFIG_PATH = "src/otx/recipes/stages/detection/incremental.py"
DEFAULT_ISEG_RECIPE_CONFIG_PATH = "src/otx/recipes/stages/instance-segmentation/incremental.py"


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
    for idx in range(number_of_images):
        if idx < 30:
            subset = Subset.VALIDATION
        else:
            subset = Subset.TRAINING
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
        items.append(DatasetItemEntityWithID(media=image, annotation_scene=annotation_scene, subset=subset))
    dataset = DatasetEntity(items)
    return dataset, dataset.get_labels()


def generate_labels(length: int, domain: Domain) -> List[LabelEntity]:
    """Generate list of LabelEntity given length and domain."""

    output: List[LabelEntity] = []
    for i in range(length):
        output.append(LabelEntity(name=f"{i + 1}", domain=domain, id=ID(i + 1)))
    return output


def create_dummy_coco_json(json_name):
    image = {
        "id": 0,
        "width": 640,
        "height": 640,
        "file_name": "fake_name.jpg",
    }

    annotation_1 = {
        "id": 1,
        "image_id": 0,
        "category_id": 0,
        "area": 400,
        "bbox": [50, 60, 20, 20],
        "segmentation": [[165.16, 2.58, 344.95, 41.29, 27.5, 363.0, 9.46, 147.1]],
        "iscrowd": 0,
    }

    annotation_2 = {
        "id": 2,
        "image_id": 0,
        "category_id": 0,
        "area": 900,
        "bbox": [100, 120, 30, 30],
        "segmentation": [[165.16, 2.58, 344.95, 41.29, 27.5, 363.0, 9.46, 147.1]],
        "iscrowd": 0,
    }

    categories = [
        {
            "id": 0,
            "name": "car",
            "supercategory": "car",
        }
    ]

    fake_json = {
        "images": [image],
        "annotations": [annotation_1, annotation_2],
        "categories": categories,
    }
    with open(json_name, "w") as f:
        json.dump(fake_json, f)
