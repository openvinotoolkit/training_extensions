# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.segmentation.adapters.mmseg.datasets.pipelines.loads import (
    LoadAnnotationFromOTXDataset,
)
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def label_entity(name="test label") -> LabelEntity:
    return LabelEntity(name=name, domain=Domain.SEGMENTATION)


def dataset_item() -> DatasetItemEntity:
    image: Image = Image(data=np.random.randint(low=0, high=255, size=(10, 16, 3)))
    annotation: Annotation = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(label_entity())])
    annotation_scene: AnnotationSceneEntity = AnnotationSceneEntity(
        annotations=[annotation], kind=AnnotationSceneKind.ANNOTATION
    )
    return DatasetItemEntity(media=image, annotation_scene=annotation_scene)


class TestLoadAnnotationFromOTXDataset:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:

        self.dataset_item: DatasetItemEntity = dataset_item()
        self.results: dict = {
            "dataset_item": self.dataset_item,
            "ann_info": {"labels": [label_entity("class_1")]},
            "seg_fields": [],
        }
        self.pipeline: LoadAnnotationFromOTXDataset = LoadAnnotationFromOTXDataset()

    @e2e_pytest_unit
    def test_call(self) -> None:
        loaded_annotations: dict = self.pipeline(self.results)
        assert "gt_semantic_seg" in loaded_annotations
        assert loaded_annotations["dataset_item"] == self.dataset_item
