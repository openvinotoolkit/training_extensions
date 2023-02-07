"""Unit Test for otx.algorithms.action.adapters.mmaction.data.pipeline.loading.."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.action.adapters.mmaction.data.cls_dataset import OTXActionClsDataset
from otx.algorithms.action.adapters.mmaction.data.pipelines.loading import (
    RawFrameDecode,
)
from otx.algorithms.action.configs.classification.x3d.data_pipeline import (
    train_pipeline,
)
from otx.api.entities.label import Domain
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.action.test_helpers import (
    MockPipeline,
    generate_action_cls_otx_dataset,
    generate_labels,
)


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
        self.labels = generate_labels(3, Domain.ACTION_CLASSIFICATION)
        self.otx_dataset = generate_action_cls_otx_dataset(self.video_len, self.frame_len, self.labels)
        self.pipeline = train_pipeline
        self.dataset = OTXActionClsDataset(self.otx_dataset, self.labels, self.pipeline)
        self.dataset.pipeline = MockPipeline()

    @e2e_pytest_unit
    def test_call(self):
        """Test __call__ function."""

        inputs = self.dataset[0]
        inputs["frame_inds"] = list(range(2))
        inputs["gt_bboxes"] = np.array([[0, 0, 1, 1]])
        inputs["proposals"] = np.array([[0, 0, 1, 1]])
        decode = RawFrameDecode()
        decode.otx_dataset = self.otx_dataset
        outputs = decode(inputs)
        assert len(outputs["imgs"]) == 2
        assert outputs["original_shape"] == (256, 256)
        assert outputs["img_shape"] == (256, 256)
        assert np.all(outputs["gt_bboxes"] == np.array([[0, 0, 256, 256]]))
        assert np.all(outputs["proposals"] == np.array([[0, 0, 256, 256]]))
