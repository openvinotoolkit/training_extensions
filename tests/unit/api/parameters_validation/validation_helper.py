"""
Common functions for input parameters validation tests
"""

import numpy as np
import pytest

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset


def check_value_error_exception_raised(correct_parameters: dict, unexpected_values: list, class_or_function) -> None:
    """
    Function checks that ValueError exception is raised when unexpected type values are specified as parameters for
    methods or functions
    """
    for key, value in unexpected_values:
        incorrect_parameters_dict = dict(correct_parameters)
        incorrect_parameters_dict[key] = value
        with pytest.raises(ValueError):
            class_or_function(**incorrect_parameters_dict)


def load_test_dataset():
    """Function prepares DatasetEntity object and labels list"""

    def gen_image(resolution, x1, y1, x2, y2):
        width, height = resolution
        image = np.full([height, width, 3], fill_value=255, dtype=np.uint8)
        image[int(y1 * height) : int(y2 * height), int(x1 * width) : int(x2 * width), :] = np.array(
            [0, 128, 128], dtype=np.uint8
        )[None, None, :]
        return image, Rectangle(x1=x1, y1=y1, x2=x2, y2=y2)

    images = [
        gen_image((640, 480), 0.0, 0.0, 0.5, 0.5),
        gen_image((640, 480), 0.5, 0.0, 1.0, 0.5),
        gen_image((640, 480), 0.0, 0.5, 0.5, 1.0),
        gen_image((640, 480), 0.5, 0.5, 1.0, 1.0),
    ]
    labels = [LabelEntity(name="rect", domain=Domain.DETECTION, id=ID("0"))]

    def get_image(i, subset):
        image, bbox = images[i]
        return DatasetItemEntity(
            media=Image(data=image),
            annotation_scene=AnnotationSceneEntity(
                annotations=[Annotation(bbox, labels=[ScoredLabel(label=labels[0])])],
                kind=AnnotationSceneKind.ANNOTATION,
            ),
            subset=subset,
        )

    items = [
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(2, Subset.TRAINING),
        get_image(3, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(2, Subset.TRAINING),
        get_image(3, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(2, Subset.VALIDATION),
        get_image(3, Subset.VALIDATION),
        get_image(0, Subset.TESTING),
        get_image(1, Subset.TESTING),
        get_image(2, Subset.TESTING),
        get_image(3, Subset.TESTING),
    ]
    return DatasetEntity(items), labels
