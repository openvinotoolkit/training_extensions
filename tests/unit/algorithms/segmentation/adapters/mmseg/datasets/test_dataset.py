# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.segmentation.adapters.mmseg.datasets import MPASegDataset
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
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def label_entity(name="test label", id="0") -> LabelEntity:
    return LabelEntity(name=name, id=ID(id), domain=Domain.SEGMENTATION)


def dataset_item() -> DatasetItemEntity:
    image: Image = Image(data=np.random.randint(low=0, high=255, size=(10, 16, 3)))
    annotation: Annotation = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(label_entity())])
    annotation_scene: AnnotationSceneEntity = AnnotationSceneEntity(
        annotations=[annotation], kind=AnnotationSceneKind.ANNOTATION
    )
    return DatasetItemEntity(media=image, annotation_scene=annotation_scene)


class TestMPASegDataset:
    @pytest.fixture(autouse=True)
    def setUp(self, mocker) -> None:
        self.otx_dataset: DatasetEntity = DatasetEntity(items=[dataset_item()])
        self.pipeline: list[dict] = [{"type": "LoadImageFromOTXDataset", "to_float32": True}]
        self.classes: list[str] = ["class_1", "class_2"]
        labels_entities = [label_entity(name, i) for i, name in enumerate(self.classes)]

        mocker.patch.object(MPASegDataset, "filter_labels", return_value=labels_entities)

        self.dataset: MPASegDataset = MPASegDataset(
            otx_dataset=self.otx_dataset,
            pipeline=self.pipeline,
            labels=labels_entities,
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
    def test_classes_sorted(self, mocker) -> None:
        self.otx_dataset: DatasetEntity = DatasetEntity(items=[dataset_item()])
        self.pipeline: list[dict] = [{"type": "LoadImageFromOTXDataset", "to_float32": True}]
        self.classes: list[str] = [f"class_{i+1}" for i in range(11)]
        labels_entities = [label_entity(name, str(i)) for i, name in enumerate(self.classes)]

        mocker.patch.object(MPASegDataset, "filter_labels", return_value=labels_entities)

        self.dataset: MPASegDataset = MPASegDataset(
            otx_dataset=self.otx_dataset,
            pipeline=self.pipeline,
            labels=labels_entities,
            new_classes=self.classes,
        )

        assert self.dataset.CLASSES == ["background"] + self.classes
        assert self.dataset.CLASSES == ["background"] + [
            label.name for label in sorted(labels_entities, key=lambda x: int(x.id))
        ]

        assert self.dataset.label_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}

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
