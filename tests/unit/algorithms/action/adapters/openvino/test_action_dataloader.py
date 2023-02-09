"""Unit Test for otx.algorithms.action.adapters.openvino.dataloader."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Optional

import numpy as np
import pytest

from otx.algorithms.action.adapters.openvino.dataloader import (
    ActionOVClsDataLoader,
    ActionOVDemoDataLoader,
    ActionOVDetDataLoader,
    _is_multi_video,
    get_ovdataloader,
)
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
from otx.api.entities.metadata import MetadataItemEntity, VideoMetadata
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockDatasetEntity(DatasetEntity):
    """Mock class for DatasetEntity."""

    def __init__(self):
        pass


class MockDatasetItemEntity(DatasetItemEntity):
    """Mock class for DatasetItemEntity"""

    def __init__(
        self,
        metadata: List[MetadataItemEntity],
        annotation_scene: Optional[AnnotationSceneEntity] = None,
    ):
        self.__metadata = metadata
        self.__annotation_scene: AnnotationSceneEntity = annotation_scene

    def get_metadata(self) -> List[MetadataItemEntity]:
        return self.__metadata


class MockImage(Image):
    """Mock class for Image entity."""

    def __init__(self):
        self.__data = np.ndarray((256, 256))
        super().__init__(self.__data)


@e2e_pytest_unit
def test_get_ovdataloader(mocker) -> None:
    """Test get_ovdataloader function.
    <Steps>
        1. Check ovdataloader type when function get ACTION_CLASSIFIACTION as task_type
        2. Check ovdataloader type when function get ACTION_DETECTION as task_type
        3. Check ovdataloader type when function get single video dataset
    """

    class MockActionOVClsDataLoader(ActionOVClsDataLoader):
        """Mock class for ActionOVClsDataLoader."""

        def __init__(self, dataset: DatasetEntity, clip_len: int, width: int, height: int):
            pass

    class MockActionOVDetDataLoader(ActionOVDetDataLoader):
        """Mock class for ActionOVDetDataLoader."""

        def __init__(self, dataset: DatasetEntity, clip_len: int, width: int, height: int):
            pass

    mocker.patch("otx.algorithms.action.adapters.openvino.dataloader._is_multi_video", return_value=True)
    mocker.patch(
        "otx.algorithms.action.adapters.openvino.dataloader.ActionOVClsDataLoader",
        side_effect=MockActionOVClsDataLoader,
    )
    mocker.patch(
        "otx.algorithms.action.adapters.openvino.dataloader.ActionOVDetDataLoader",
        side_effect=MockActionOVDetDataLoader,
    )

    _task_type = "ACTION_CLASSIFICATION"
    out = get_ovdataloader(MockDatasetEntity(), _task_type, 8, 256, 256)
    assert isinstance(out, ActionOVClsDataLoader)

    _task_type = "ACTION_DETECTION"
    out = get_ovdataloader(MockDatasetEntity(), _task_type, 8, 256, 256)
    assert isinstance(out, ActionOVDetDataLoader)

    _task_type = "ACTION_SEGMENTATION"
    with pytest.raises(NotImplementedError):
        out = get_ovdataloader(MockDatasetEntity, _task_type, 8, 256, 256)

    mocker.patch("otx.algorithms.action.adapters.openvino.dataloader._is_multi_video", return_value=False)

    _task_type = "ACTION_DETECTION"
    out = get_ovdataloader(MockDatasetEntity(), _task_type, 8, 256, 256)
    assert isinstance(out, ActionOVDemoDataLoader)


@e2e_pytest_unit
def test_is_multi_video() -> None:
    """Test _is_multi_video function.

    <Steps>
        1. Check return value: bool when function get single video DatasetEntity
        2. Check return value: bool when function get multi video DatasetEntity
    """

    items: List[DatasetItemEntity] = []
    items.append(MockDatasetItemEntity(metadata=[MetadataItemEntity(data=VideoMetadata("2", 0, False))]))
    items.append(MockDatasetItemEntity(metadata=[MetadataItemEntity(data=VideoMetadata("2", 1, False))]))
    items.append(MockDatasetItemEntity(metadata=[MetadataItemEntity(data=VideoMetadata("2", 2, False))]))

    dataset = DatasetEntity(items)
    assert _is_multi_video(dataset) is False

    items.append(MockDatasetItemEntity(metadata=[MetadataItemEntity(data=VideoMetadata("1", 0, False))]))
    items.append(MockDatasetItemEntity(metadata=[MetadataItemEntity(data=VideoMetadata("1", 1, False))]))
    items.append(MockDatasetItemEntity(metadata=[MetadataItemEntity(data=VideoMetadata("1", 2, False))]))

    dataset = DatasetEntity(items)
    assert _is_multi_video(dataset) is True


class TestActionOVDemoDataLoader:
    """Test ActionOVDemoDataLoader class.

    1. Initialize ActionOVDemoDataLoader and check its length
    2. Test __getitem__ function
    <Steps>
        1. Create ActionOVDemoDataLoader
        2. Sample first item from ActionOVDemoDataLoader
        3. The item's frame indices should be [1, 1, 1, 1, 1, 3, 5, 7]
    3. Test add_prediction function
    <Steps>
        1. Create sample prediction
        2. Check whether empty annotation changed to sample prediction
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.data_len = 40
        self.items: List[DatasetItemEntity] = []
        for i in range(self.data_len):
            self.items.append(
                DatasetItemEntity(
                    media=MockImage(),
                    annotation_scene=AnnotationSceneEntity(
                        annotations=[
                            Annotation(
                                Rectangle.generate_full_box(),
                                [ScoredLabel(LabelEntity("1", Domain.ACTION_CLASSIFICATION, id=ID(1)))],
                            )
                        ],
                        kind=AnnotationSceneKind.ANNOTATION,
                    ),
                    metadata=[MetadataItemEntity(data=VideoMetadata("1", i + 1, False))],
                )
            )
        self.dataset = DatasetEntity(self.items)

    @e2e_pytest_unit
    def test_len(self) -> None:
        """Test initialization and __len__ function."""

        dataloader = ActionOVDemoDataLoader(self.dataset, "ACTION_CLASSIFICATION", 8, 256, 256)
        assert len(dataloader) == self.data_len

    @e2e_pytest_unit
    def test_getitem(self) -> None:
        """Test __getitem__ function."""

        dataloader = ActionOVDemoDataLoader(self.dataset, "ACTION_CLASSIFICATION", 8, 256, 256)
        outs = dataloader[0]
        frame_indices: List[int] = []
        for out in outs:
            frame_idx = out.get_metadata()[0].data.frame_idx
            frame_indices.append(frame_idx)
        assert frame_indices == [1, 1, 1, 1, 1, 3, 5, 7]

    @e2e_pytest_unit
    def test_add_prediction(self) -> None:
        """Test add_prediction function."""
        prediction = AnnotationSceneEntity(
            annotations=[
                Annotation(
                    Rectangle.generate_full_box(),
                    [ScoredLabel(LabelEntity("2", Domain.ACTION_CLASSIFICATION, id=ID(1)))],
                )
            ],
            kind=AnnotationSceneKind.ANNOTATION,
        )

        dataloader = ActionOVDemoDataLoader(self.dataset, "ACTION_CLASSIFICATION", 8, 256, 256)
        items = self.dataset.with_empty_annotations()._items
        dataloader.add_prediction(items, prediction)
        assert len(items[0].get_annotations()) == 0
        assert len(items[len(items) // 2].get_annotations()) >= 1

        dataloader = ActionOVDemoDataLoader(self.dataset, "ACTION_DETECTION", 8, 256, 256)
        items = self.dataset.with_empty_annotations()._items
        dataloader.add_prediction(items, prediction)
        assert len(items[0].get_annotations()) == 0
        assert len(items[len(items) // 2].get_annotations()) >= 1


class TestActionOVClsDataLoader:
    """Test class for ActionOVClsDataLoader.

    1. Test initialization
    <Steps>
        1. Check self.dataloader's length. It should be 1 because all dataset item have same video_id
        2. Check self.dataloader.dataset's length. It should be self.data_len(40)
    2. Test __getitem__
    <Steps>
        1. Check len(output) == clip_len(8)
        2. Check frame_indices == [5, 9, 13, 17, 21, 25, 29, 33]. It comes from setting indices rule.
    3. Test add_prediciton
    <Steps>
        1. Check self.dataset.get_labels(). It should be 1, because only one label is added to empty dataset.
        2. Check self.dataset's label's id is 2.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.data_len = 40
        self.items: List[DatasetItemEntity] = []
        for i in range(self.data_len):
            self.items.append(
                DatasetItemEntity(
                    media=MockImage(),
                    annotation_scene=AnnotationSceneEntity(
                        annotations=[
                            Annotation(
                                Rectangle.generate_full_box(),
                                [ScoredLabel(LabelEntity("1", Domain.ACTION_CLASSIFICATION, id=ID(1)))],
                            )
                        ],
                        kind=AnnotationSceneKind.ANNOTATION,
                    ),
                    metadata=[MetadataItemEntity(data=VideoMetadata("1", i + 1, False))],
                )
            )
        self.dataset = DatasetEntity(self.items)
        self.dataloader = ActionOVClsDataLoader(self.dataset, 8, 256, 256)

    @e2e_pytest_unit
    def test_init(self) -> None:
        """Test __init__ function."""

        assert len(self.dataloader) == 1
        assert len(self.dataloader.dataset[0]) == self.data_len

    @e2e_pytest_unit
    def test_getitem(self) -> None:
        """Test __getitem__ function."""

        outs = self.dataloader[0]
        frame_indices = []
        for out in outs:
            frame_idx = out.get_metadata()[0].data.frame_idx
            frame_indices.append(frame_idx)
        assert len(outs) == 8
        assert frame_indices == [5, 9, 13, 17, 21, 25, 29, 33]

    @e2e_pytest_unit
    def test_add_prediction(self) -> None:
        """Test add_prediciton function."""

        prediction = AnnotationSceneEntity(
            annotations=[
                Annotation(
                    Rectangle.generate_full_box(),
                    [ScoredLabel(LabelEntity("2", Domain.ACTION_CLASSIFICATION, id=ID(2)))],
                )
            ],
            kind=AnnotationSceneKind.ANNOTATION,
        )
        self.dataset = self.dataset.with_empty_annotations()
        self.dataloader.add_prediction(self.dataset, self.items, prediction)
        assert len(self.dataset.get_labels()) == 1
        assert int(self.dataset.get_labels()[0].id) == 2


class TestActionOVDetDataLoader:
    """Test class for ActionOVDetDataLoader.

    1. Test initialization
    <Steps>
        1. Check self.dataloader's length. It should be 1 because all dataset item have same video_id
        2. Check self.dataloader.dataset's length. It should be self.data_len(40)
    2. Test __getitem__
    <Steps>
        1. Check len(output) == clip_len(8)
        2. Check frame_indices == [5, 9, 13, 17, 21, 25, 29, 33]. It comes from setting indices rule.
    3. Test add_prediciton
    <Steps>
        1. Check self.dataset.get_labels(). It should be 1, because only one label is added to empty dataset.
        2. Check self.dataset's label's id is 2.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.data_len = 40
        self.items: List[DatasetItemEntity] = []
        for i in range(self.data_len):
            if i % 5 == 0:
                metadata = MetadataItemEntity(data=VideoMetadata("1", i + 1, True))
            else:
                metadata = MetadataItemEntity(data=VideoMetadata("1", i + 1, False))
            self.items.append(
                DatasetItemEntity(
                    media=MockImage(),
                    annotation_scene=AnnotationSceneEntity(
                        annotations=[
                            Annotation(
                                Rectangle.generate_full_box(),
                                [ScoredLabel(LabelEntity("1", Domain.ACTION_DETECTION, id=ID(1)))],
                            )
                        ],
                        kind=AnnotationSceneKind.ANNOTATION,
                    ),
                    metadata=[metadata],
                )
            )
        self.dataset = DatasetEntity(self.items)
        self.dataloader = ActionOVDetDataLoader(self.dataset, 8, 256, 256)

    @e2e_pytest_unit
    def test_init(self) -> None:
        """Test __init__ function."""

        assert len(self.dataloader) == 32
        assert len(self.dataloader.original_dataset) == 40

        sample = self.dataloader.dataset[0]
        metadata = sample.get_metadata()[0].data
        assert "start_index" in metadata.metadata
        assert "timestamp_start" in metadata.metadata
        assert "timestamp_end" in metadata.metadata

    @e2e_pytest_unit
    def test_getitem(self) -> None:
        """Test __getitem__ function."""

        outs = self.dataloader[0]
        frame_indices = []
        for out in outs:
            frame_idx = out.get_metadata()[0].data.frame_idx
            frame_indices.append(frame_idx)
        assert len(outs) == 8
        assert frame_indices == [1, 1, 1, 1, 2, 4, 6, 8]

    @e2e_pytest_unit
    def test_add_prediction(self) -> None:
        """Test add_prediciton function."""

        prediction = AnnotationSceneEntity(
            annotations=[
                Annotation(
                    Rectangle.generate_full_box(), [ScoredLabel(LabelEntity("2", Domain.ACTION_DETECTION, id=ID(2)))]
                )
            ],
            kind=AnnotationSceneKind.ANNOTATION,
        )
        items = self.dataset.with_empty_annotations()._items
        self.dataloader.add_prediction(items, prediction)
        assert len(items[0].get_annotations()) == 0
        assert len(items[len(items) // 2].get_annotations()) >= 1
