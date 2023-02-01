"""Unit Test for otx.algorithms.action.adapters.mmaction.data.pipeline.loading.."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import List

import numpy as np
import pytest

from otx.algorithms.action.adapters.mmaction.data.cls_dataset import OTXActionClsDataset
from otx.algorithms.action.adapters.mmaction.data.pipelines.loading import (
    RawFrameDecode,
)
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
from tests.unit.algorithms.action.adapters.mmaction.data.test_action_cls_dataset import (
    MockPipeline,
)


class MockImage(Image):
    """Mock class for Image entity."""

    @property
    def numpy(self) -> np.ndarray:
        """Returns empty numpy array"""
        return np.ndarray((256, 256))


class TestRawFrameDecode:
    """Test RawFrameDecode class.

    <Steps>
        1. Create sample OTXActionClsDataset
        2. Get sample inputs from sample OTXActionClsDataset
        3. Add "frame_inds", "gt_bboxes", "proposals" attributes to sample inputs
        4. Check RawFrameDecode transform's results
            1. Whether transform creates imgs
            2. Whether transform creates proper img size
            3. Whether transform modify gt_bboxes and proposals w.r.t img_size
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
                    media=MockImage(f"{video_id}_{frame_idx}.png"),
                    annotation_scene=AnnotationSceneEntity(
                        annotations=[Annotation(Rectangle.generate_full_box(), [ScoredLabel(self.labels[video_id])])],
                        kind=AnnotationSceneKind.ANNOTATION,
                    ),
                    metadata=[MetadataItemEntity(data=VideoMetadata(video_id, frame_idx, is_empty_frame=False))],
                )
                items.append(item)
        self.otx_dataset = DatasetEntity(items=items)
        self.pipeline = train_pipeline
        self.dataset = OTXActionClsDataset(self.otx_dataset, self.labels, self.pipeline)
        self.dataset.pipeline = MockPipeline()
        self.decode = RawFrameDecode()
        self.decode.otx_dataset = self.otx_dataset

    @e2e_pytest_unit
    def test_call(self):
        """Test __call__ function."""

        inputs = self.dataset[0]
        inputs["frame_inds"] = list(range(2))
        inputs["gt_bboxes"] = np.array([[0, 0, 1, 1]])
        inputs["proposals"] = np.array([[0, 0, 1, 1]])
        outputs = self.decode(inputs)
        assert len(outputs["imgs"]) == 2
        assert outputs["original_shape"] == (256, 256)
        assert outputs["img_shape"] == (256, 256)
        assert np.all(outputs["gt_bboxes"] == np.array([[0, 0, 256, 256]]))
        assert np.all(outputs["proposals"] == np.array([[0, 0, 256, 256]]))
