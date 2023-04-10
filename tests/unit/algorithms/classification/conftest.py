# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntityWithID
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle


@pytest.fixture()
def fxt_multi_class_cls_dataset_entity() -> DatasetEntity:
    labels = ["car", "dog", "cat"]
    items = []
    for label in labels:
        image = Image(data=np.random.randint(low=0, high=255, size=(8, 8, 3)))
        annotation = Annotation(
            shape=Rectangle.generate_full_box(),
            labels=[ScoredLabel(LabelEntity(name=label, domain=Domain.CLASSIFICATION, id=ID(label)))],
        )
        annotation_scene = AnnotationSceneEntity(annotations=[annotation], kind=AnnotationSceneKind.ANNOTATION)
        items += [DatasetItemEntityWithID(media=image, annotation_scene=annotation_scene, id_=ID(label))]

    dataset = DatasetEntity(items=items)
    return dataset
