"""
OTE parameters validation tests helpers
"""

import numpy as np

from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset


def load_test_dataset():
    """Helper to create test dataset"""

    def gen_image(resolution, x1, y1, x2, y2):
        w, h = resolution
        image = np.full([h, w, 3], fill_value=255, dtype=np.uint8)
        image[int(y1 * h): int(y2 * h), int(x1 * w): int(x2 * w), :] = np.array(
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
