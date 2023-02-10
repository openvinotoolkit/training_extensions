# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import PIL.Image
import pytest

from otx.algorithms.segmentation.adapters.mmseg.data.pipelines import (
    LoadAnnotationFromOTXDataset,
    NDArrayToPILImage,
    PILImageToNDArray,
    RandomResizedCrop,
    RandomSolarization,
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


class TestNDArrayToPILImage:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.results: dict = {"img": np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)}
        self.nd_array_to_pil_image: NDArrayToPILImage = NDArrayToPILImage(keys=["img"])

    @e2e_pytest_unit
    def test_call(self) -> None:
        converted_img: dict = self.nd_array_to_pil_image(self.results)
        assert "img" in converted_img
        assert isinstance(converted_img["img"], PIL.Image.Image)

    @e2e_pytest_unit
    def test_repr(self) -> None:
        assert str(self.nd_array_to_pil_image) == "NDArrayToPILImage"


class TestPILImageToNDArray:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.results: dict = {"img": PIL.Image.new("RGB", (3, 3))}
        self.pil_image_to_nd_array: PILImageToNDArray = PILImageToNDArray(keys=["img"])

    @e2e_pytest_unit
    def test_call(self) -> None:
        converted_array: dict = self.pil_image_to_nd_array(self.results)
        assert "img" in converted_array
        assert isinstance(converted_array["img"], np.ndarray)

    @e2e_pytest_unit
    def test_repr(self) -> None:
        assert str(self.pil_image_to_nd_array) == "PILImageToNDArray"


class TestRandomResizedCrop:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.results: dict = {"img": PIL.Image.new("RGB", (10, 16)), "img_shape": (10, 16), "ori_shape": (10, 16)}
        self.random_resized_crop: RandomResizedCrop = RandomResizedCrop((5, 5), (0.5, 1.0))

    @e2e_pytest_unit
    def test_call(self) -> None:
        cropped_img: dict = self.random_resized_crop(self.results)
        assert cropped_img["img_shape"] == (5, 5)
        assert cropped_img["ori_shape"] == (10, 16)


class TestRandomSolarization:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.results: dict = {"img": np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)}
        self.random_solarization: RandomSolarization = RandomSolarization(p=1.0)

    @e2e_pytest_unit
    def test_call(self) -> None:
        solarized: dict = self.random_solarization(self.results)
        assert "img" in solarized
        assert isinstance(solarized["img"], np.ndarray)

    @e2e_pytest_unit
    def test_repr(self) -> None:
        assert str(self.random_solarization) == "RandomSolarization"
