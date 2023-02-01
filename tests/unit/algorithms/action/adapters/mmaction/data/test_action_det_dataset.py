"""Unit Test for otx.algorithms.action.adapters.mmaction.data.cls_dataset.."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List

import numpy as np
import pytest

from otx.algorithms.action.adapters.mmaction.data.det_dataset import OTXActionDetDataset
from otx.algorithms.action.adapters.mmaction.data.pipelines import RawFrameDecode
from otx.algorithms.action.configs.detection.base.data_pipeline import train_pipeline
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
from tests.unit.algorithms.action.adapters.mmaction.data.test_action_cls_dataset import (
    MockPipeline,
)


class MockDataInfoProxy(OTXActionDetDataset._DataInfoProxy):
    """Mock class for _DataInfoProxy of OTXAcitonDetDataset."""

    def __init__(self, proposals):
        self.proposals = proposals


class TestOTXActionDetDataset:
    """Test OTXActionDetDataset class.

    1. Check _DataInfoProxy
    <Steps>
        1. Create otx_dataset, labels
        2. Create _DataInfoProxy
        3. Check metadata and annotations
        4. Create proposals and check _patch_proposals
    2. Check pipelines
    3. Check loading functions
    4. Check evaluation function
    <Steps>
        1. Create sample detection inference results
        2. Check det2csv function's results
        3. Check _get_predictions function's results
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.video_len = 3
        self.frame_len = 5
        self.labels = [
            LabelEntity(name="1", domain=Domain.ACTION_DETECTION, id=ID(1)),
            LabelEntity(name="2", domain=Domain.ACTION_DETECTION, id=ID(2)),
            LabelEntity(name="3", domain=Domain.ACTION_DETECTION, id=ID(3)),
        ]
        items: List[DatasetItemEntity] = []
        self.proposals: Dict[str, List[float]] = {}
        for video_id in range(self.video_len):
            for frame_idx in range(self.frame_len):
                if frame_idx % 2 == 0:
                    item = DatasetItemEntity(
                        media=Image(f"{video_id}_{frame_idx}.png"),
                        annotation_scene=AnnotationSceneEntity(
                            annotations=[
                                Annotation(Rectangle.generate_full_box(), [ScoredLabel(self.labels[video_id])])
                            ],
                            kind=AnnotationSceneKind.ANNOTATION,
                        ),
                        metadata=[
                            MetadataItemEntity(data=VideoMetadata(str(video_id), frame_idx, is_empty_frame=False))
                        ],
                    )
                    self.proposals[f"{video_id},{frame_idx:04d}"] = [0.0, 0.0, 1.0, 1.0]
                else:
                    item = DatasetItemEntity(
                        media=Image(f"{video_id}_{frame_idx}.png"),
                        annotation_scene=AnnotationSceneEntity(
                            annotations=[
                                Annotation(Rectangle.generate_full_box(), [ScoredLabel(self.labels[video_id])])
                            ],
                            kind=AnnotationSceneKind.ANNOTATION,
                        ),
                        metadata=[
                            MetadataItemEntity(data=VideoMetadata(str(video_id), frame_idx, is_empty_frame=True))
                        ],
                    )
                items.append(item)
        self.otx_dataset = DatasetEntity(items=items)
        self.pipeline = train_pipeline

    @e2e_pytest_unit
    def test_DataInfoProxy(self) -> None:
        """Test _DataInfoProxy in OTXActionDeteDataset."""

        proxy = OTXActionDetDataset._DataInfoProxy(self.otx_dataset, self.labels, fps=1)
        sample = proxy[0]
        assert len(proxy) == 9
        assert sample["shot_info"] == (0, 4)
        assert sample["is_empty_frame"] is False
        assert "start_index" in sample
        assert "video_id" in sample
        assert "timestamp" in sample
        assert "gt_bboxes" in sample
        assert "gt_labels" in sample

    @e2e_pytest_unit
    def test_DataInfoProxy_patch_proposals(self) -> None:
        """Test _patch_proposals function in _DataInfoProxy class."""

        proxy = MockDataInfoProxy(self.proposals)
        pre_len = len(proxy.proposals)
        proxy._patch_proposals()
        assert len(proxy.proposals) == pre_len
        for key in proxy.proposals:
            assert str(int(key.split(",")[-1])) == key.split(",")[-1]

    @e2e_pytest_unit
    def test_pipeline(self) -> None:
        """Test RawFrameDecode transform contains otx_dataset."""

        dataset = OTXActionDetDataset(self.otx_dataset, self.labels, self.pipeline, fps=1)
        for transform in dataset.pipeline.transforms:
            if isinstance(transform, RawFrameDecode):
                assert transform.otx_dataset == self.otx_dataset

    @e2e_pytest_unit
    def test_prepare_train_frames(self) -> None:
        """Test prepare_train_frames function.

        prepare_train_frames function's output should contain essential attributes for training
        """

        dataset = OTXActionDetDataset(self.otx_dataset, self.labels, self.pipeline, fps=1)
        dataset.pipeline = MockPipeline()
        sample = dataset.prepare_train_frames(0)
        assert sample["shot_info"] == (0, 4)
        assert sample["is_empty_frame"] is False
        assert "start_index" in sample
        assert "video_id" in sample
        assert "timestamp" in sample
        assert "gt_bboxes" in sample
        assert "gt_labels" in sample

    @e2e_pytest_unit
    def test_prepare_test_frames(self) -> None:
        """Test prepare_test_frames function.

        Same with test_prepare_train_frames
        """

        dataset = OTXActionDetDataset(self.otx_dataset, self.labels, self.pipeline, fps=1)
        dataset.pipeline = MockPipeline()
        sample = dataset.prepare_test_frames(0)
        assert sample["shot_info"] == (0, 4)
        assert sample["is_empty_frame"] is False
        assert "start_index" in sample
        assert "video_id" in sample
        assert "timestamp" in sample
        assert "gt_bboxes" in sample
        assert "gt_labels" in sample

    @e2e_pytest_unit
    def test_evaluate(self, mocker) -> None:
        """Test evaluate function"""

        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.data.det_dataset.det_eval", return_value={"mAP@0.5IOU": 0.5}
        )
        dataset = OTXActionDetDataset(self.otx_dataset, self.labels, self.pipeline, fps=1)
        results = [
            [
                np.array([[0.0, 0.0, 1.0, 1.0, 1.0]]),
                np.array([[0.0, 0.0, 1.0, 1.0, 0.0]]),
                np.array([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            ]
        ] * 9
        output = dataset.evaluate(results)
        assert output == {"mAP@0.5IOU": 0.5}
