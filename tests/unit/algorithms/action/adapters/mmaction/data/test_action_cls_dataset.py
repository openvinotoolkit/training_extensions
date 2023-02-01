"""Unit Test for otx.algorithms.action.adapters.mmaction.data.cls_dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, List

import pytest

from otx.algorithms.action.adapters.mmaction.data.cls_dataset import OTXActionClsDataset
from otx.algorithms.action.adapters.mmaction.data.pipelines import RawFrameDecode
from otx.algorithms.action.configs.classification.x3d.data_pipeline import (
    train_pipeline,
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


class MockPipeline:
    """Mock class for data pipeline.

    It returns its inputs.
    """

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return results


class TestOTXActionClsDataset:
    """Test OTXActionClsDataset class.

    1. Check _DataInfoProxy
    <Steps>
        1. Create otx_dataset, labels
        2. Check len(_DataInfoProxy)
    2. Check data pipelines
    3. Check "__len__" function
    4. Check "prepare_train_frames" function
    5. Check "prepare_test_frames" function
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.video_len = 3
        self.frame_len = 3
        self.labels = [
            LabelEntity(name="1", domain=Domain.ACTION_CLASSIFICATION, id=ID(1)),
            LabelEntity(name="2", domain=Domain.ACTION_CLASSIFICATION, id=ID(2)),
            LabelEntity(name="3", domain=Domain.ACTION_CLASSIFICATION, id=ID(3)),
        ]
        items: List[DatasetItemEntity] = []
        for video_id in range(self.video_len):
            for frame_idx in range(self.frame_len):
                item = DatasetItemEntity(
                    media=Image(f"{video_id}_{frame_idx}.png"),
                    annotation_scene=AnnotationSceneEntity(
                        annotations=[Annotation(Rectangle.generate_full_box(), [ScoredLabel(self.labels[video_id])])],
                        kind=AnnotationSceneKind.ANNOTATION,
                    ),
                    metadata=[MetadataItemEntity(data=VideoMetadata(video_id, frame_idx, is_empty_frame=False))],
                )
                items.append(item)
        self.otx_dataset = DatasetEntity(items=items)
        self.pipeline = train_pipeline

    @e2e_pytest_unit
    def test_DataInfoProxy(self) -> None:
        """Test _DataInfoProxy Class."""

        proxy = OTXActionClsDataset._DataInfoProxy(self.otx_dataset, self.labels, modality="RGB")
        sample = proxy[0]
        assert len(proxy) == self.video_len
        assert "total_frames" in sample
        assert "start_index" in sample
        assert "label" in sample

    @e2e_pytest_unit
    def test_pipeline(self) -> None:
        """Test RawFrameDecode transform contains otx_dataset."""

        dataset = OTXActionClsDataset(self.otx_dataset, self.labels, self.pipeline)
        for transform in dataset.pipeline.transforms:
            if isinstance(transform, RawFrameDecode):
                assert transform.otx_dataset == self.otx_dataset

    @e2e_pytest_unit
    def test_len(self) -> None:
        """Test dataset length is same with video_len."""

        dataset = OTXActionClsDataset(self.otx_dataset, self.labels, self.pipeline)
        assert len(dataset) == self.video_len

    @e2e_pytest_unit
    def test_prepare_train_frames(self) -> None:
        """Test prepare_train_frames function."""

        dataset = OTXActionClsDataset(self.otx_dataset, self.labels, self.pipeline)
        dataset.pipeline = MockPipeline()
        sample = dataset.prepare_train_frames(0)
        assert "total_frames" in sample
        assert "start_index" in sample
        assert "label" in sample

    @e2e_pytest_unit
    def test_prepare_test_frames(self) -> None:
        """Test prepare_test_frames function."""

        dataset = OTXActionClsDataset(self.otx_dataset, self.labels, self.pipeline)
        dataset.pipeline = MockPipeline()
        sample = dataset.prepare_test_frames(0)
        assert "total_frames" in sample
        assert "start_index" in sample
        assert "label" in sample
