# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.segmentation.adapters.mmseg.data.dataset import MPASegDataset
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
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


class TestMPASegDataset:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.otx_dataset: DatasetEntity = DatasetEntity(items=[dataset_item()])
        self.pipeline: list[dict] = [{"type": "LoadImageFromOTXDataset", "to_float32": True}]
        self.classes: list[str] = ["class_1", "class_2"]

        self.dataset: MPASegDataset = MPASegDataset(
            otx_dataset=self.otx_dataset,
            pipeline=self.pipeline,
            labels=[label_entity(name) for name in self.classes],
            new_classes=self.classes,
        )

    @e2e_pytest_unit
    def test_mpasegdataset_initialization(self) -> None:
        assert self.dataset.otx_dataset == self.otx_dataset
        assert self.dataset.CLASSES == ["background"] + self.classes

        # Check if img_indices are generated properly
        assert hasattr(self.dataset, "img_indices")

        # Check if label_map is created as expected
        assert self.dataset.label_map == {0: 0, 1: 1, 2: 2}

    @e2e_pytest_unit
    def test_getitem_method(self) -> None:
        data_item: dict = self.dataset[0]
        assert "dataset_item" in data_item
        assert "ann_info" in data_item

    @e2e_pytest_unit
    def test_getting_annotation_info(self) -> None:
        annotation_info: dict = self.dataset.get_ann_info(0)
        assert "gt_semantic_seg" in annotation_info

    @e2e_pytest_unit
    def test_get_gt_seg_maps(self) -> None:
        gt_seg_map: np.ndarray = self.dataset.get_gt_seg_maps()[0]
        assert np.equal(gt_seg_map, np.zeros((10, 16))).all()
