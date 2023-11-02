"""Collection of helper functions for unit tests of otx.algorithms.detection."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import logging
import random
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np

from otx.algorithms.detection.utils import generate_label_schema
from otx.v2.api.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from otx.v2.api.entities.dataset_item import DatasetItemEntityWithID
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.id import ID
from otx.v2.api.entities.image import Image
from otx.v2.api.entities.label import Domain, LabelEntity
from otx.v2.api.entities.task_type import TaskType
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.scored_label import ScoredLabel
from otx.v2.api.entities.shapes.ellipse import Ellipse
from otx.v2.api.entities.shapes.polygon import Point, Polygon
from otx.v2.api.entities.shapes.rectangle import Rectangle
from otx.v2.api.entities.utils.shape_factory import ShapeFactory


logger = logging.getLogger(__name__)


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


def generate_det_dataset(task_type, number_of_images=1, image_width=640, image_height=480):
    classes = ("rectangle", "ellipse", "triangle")
    label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))

    items = []
    for idx in range(number_of_images):
        if idx < 30:
            subset = Subset.VALIDATION
        else:
            subset = Subset.TRAINING
        image_numpy, annos = generate_random_annotated_image(
            image_width=image_width,
            image_height=image_height,
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


def task_type_to_label_domain(task_type: TaskType) -> Domain:
    """Return proper label domain given task type."""
    if task_type == TaskType.DETECTION:
        return Domain.DETECTION
    if task_type == TaskType.INSTANCE_SEGMENTATION:
        return Domain.INSTANCE_SEGMENTATION
    return Domain.ROTATED_DETECTION


def generate_labels(length: int, domain: Domain) -> List[LabelEntity]:
    """Generate list of LabelEntity given length and domain."""

    output: List[LabelEntity] = []
    for i in range(length):
        output.append(LabelEntity(name=f"{i + 1}", domain=domain, id=ID(i + 1)))
    return output


def generate_random_annotated_image(
    image_width: int,
    image_height: int,
    labels: Sequence[LabelEntity],
    min_size=50,
    max_size=250,
    shape: Optional[str] = None,
    max_shapes: int = 10,
    intensity_range: List[Tuple[int, int]] = None,
    random_seed: Optional[int] = None,
    use_mask_as_annotation: bool = False,
) -> Tuple[np.ndarray, List[Annotation]]:
    """
    Generate a random image with the corresponding annotation entities.

    :param intensity_range: Intensity range for RGB channels ((r_min, r_max), (g_min, g_max), (b_min, b_max))
    :param max_shapes: Maximum amount of shapes in the image
    :param shape: {"rectangle", "ellipse", "triangle"}
    :param image_height: Height of the image
    :param image_width: Width of the image
    :param labels: Task Labels that should be applied to the respective shape
    :param min_size: Minimum size of the shape(s)
    :param max_size: Maximum size of the shape(s)
    :param random_seed: Seed to initialize the random number generator
    :param use_mask_as_annotation: If True, masks will be added in annotation
    :return: uint8 array, list of shapes
    """
    from skimage.draw import random_shapes, rectangle

    if intensity_range is None:
        intensity_range = [(100, 200)]

    image1: Optional[np.ndarray] = None
    sc_labels = []
    # Sporadically, it might happen there is no shape in the image, especially on low-res images.
    # It'll retry max 5 times until we see a shape, and otherwise raise a runtime error
    if shape == "ellipse":  # ellipse shape is not available in random_shapes function. use circle instead
        shape = "circle"
    for _ in range(5):
        rand_image, sc_labels = random_shapes(
            (image_height, image_width),
            min_shapes=1,
            max_shapes=max_shapes,
            intensity_range=intensity_range,
            min_size=min_size,
            max_size=max_size,
            shape=shape,
            random_seed=random_seed,
        )
        num_shapes = len(sc_labels)
        if num_shapes > 0:
            image1 = rand_image
            break

    if image1 is None:
        raise RuntimeError("Was not able to generate a random image that contains any shapes")

    annotations: List[Annotation] = []
    for sc_label in sc_labels:
        sc_label_name = sc_label[0]
        sc_label_shape_r = sc_label[1][0]
        sc_label_shape_c = sc_label[1][1]
        y_min, y_max = max(0.0, float(sc_label_shape_r[0] / image_height)), min(
            1.0, float(sc_label_shape_r[1] / image_height)
        )
        x_min, x_max = max(0.0, float(sc_label_shape_c[0] / image_width)), min(
            1.0, float(sc_label_shape_c[1] / image_width)
        )

        if sc_label_name == "ellipse":
            # Fix issue with newer scikit-image libraries that generate ellipses.
            # For now we render a rectangle on top of it
            sc_label_name = "rectangle"
            rr, cc = rectangle(
                start=(sc_label_shape_r[0], sc_label_shape_c[0]),
                end=(sc_label_shape_r[1] - 1, sc_label_shape_c[1] - 1),
                shape=image1.shape,
            )
            image1[rr, cc] = (
                # disable B311 random - used for the random sampling not for security/crypto
                random.randint(0, 200),  # nosec B311
                random.randint(0, 200),  # nosec B311
                random.randint(0, 200),  # nosec B311
            )
        if sc_label_name == "circle":
            sc_label_name = "ellipse"

        label_matches = [label for label in labels if sc_label_name == label.name]
        if len(label_matches) > 0:
            label = label_matches[0]
            box_annotation = Annotation(
                Rectangle(x1=x_min, y1=y_min, x2=x_max, y2=y_max),
                labels=[ScoredLabel(label, probability=1.0)],
            )

            annotation: Annotation

            if label.name == "ellipse":
                annotation = Annotation(
                    Ellipse(
                        x1=box_annotation.shape.x1,
                        y1=box_annotation.shape.y1,
                        x2=box_annotation.shape.x2,
                        y2=box_annotation.shape.y2,
                    ),
                    labels=box_annotation.get_labels(include_empty=True),
                )
            elif label.name == "triangle":
                points = [
                    Point(
                        x=(box_annotation.shape.x1 + box_annotation.shape.x2) / 2,
                        y=box_annotation.shape.y1,
                    ),
                    Point(x=box_annotation.shape.x1, y=box_annotation.shape.y2),
                    Point(x=box_annotation.shape.x2, y=box_annotation.shape.y2),
                ]

                annotation = Annotation(
                    Polygon(points=points),
                    labels=box_annotation.get_labels(include_empty=True),
                )
            else:
                annotation = box_annotation

            annotations.append(annotation)

            if use_mask_as_annotation:
                mask = np.zeros_like(image1, dtype=np.uint8)
                y_min, y_max = int(y_min * image_height), int(y_max * image_height)
                x_min, x_max = int(x_min * image_width), int(x_max * image_width)

                coords_object = np.where(image1[y_min:y_max, x_min:x_max] < 255)
                mask[y_min:y_max, x_min:x_max][coords_object] = 1
                mask = mask.sum(axis=-1)
                mask[mask > 0] = 1
                mask_annotation = Annotation(
                    Image(data=mask, size=mask.shape),
                    labels=box_annotation.get_labels(include_empty=True),
                )
                annotations.append(mask_annotation)
        else:
            logger.warning(
                "Generated a random image, but was not able to associate a label with a shape. "
                f"The name of the shape was `{sc_label_name}`. "
            )

    return image1, annotations


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
