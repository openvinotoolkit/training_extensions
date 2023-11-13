"""Unit Test for otx.v2.algorithms.action.adapters.mmaction.data.cls_dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from mmaction.utils import register_all_modules
from otx.v2.adapters.torch.mmengine.mmaction.modules.datasets.otx_action_cls_dataset import OTXActionClsDataset
from mmaction.datasets.transforms import RawFrameDecode
from otx.v2.adapters.torch.mmengine.mmaction.dataset import (
    get_default_pipeline,
)
from otx.api.entities.label import Domain
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.action.test_helpers import (
    generate_action_cls_otx_dataset,
    generate_labels,
)


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
        register_all_modules(init_default_scope=True)
        self.video_len = 3
        self.frame_len = 3
        self.labels = generate_labels(3, Domain.ACTION_CLASSIFICATION)
        self.otx_dataset = generate_action_cls_otx_dataset(self.video_len, self.frame_len, self.labels)
        self.pipeline = get_default_pipeline(subset="test")

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
