"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.adaptive_repeat_data_hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from mmcv.utils import Config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.common.adapters.mmcv.hooks import AdaptiveRepeatDataHook


class TestAdaptiveRepeatDataHook:
    """Test class for AdaptiveRepeatDataHook."""

    @pytest.fixture(autouse=True)
    def setup(self):
        class MockDataset(Dataset):
            def __init__(self):
                self.img_indices = {"foo": list(range(0, 6)), "bar": list(range(6, 10))}

            def __len__(self):
                return 10

        self.mock_dataset = MockDataset()
        self.mock_data_loader = DataLoader(
            dataset=MockDataset(),
            batch_size=len(MockDataset()),
        )
        self.mock_runner = Config({"data_loader": self.mock_data_loader})

    @e2e_pytest_unit
    def test_before_epoch(self) -> None:
        hook = AdaptiveRepeatDataHook(64, len(self.mock_dataset))
        hook.before_epoch(self.mock_runner)

        assert self.mock_runner.data_loader.sampler.repeat == 5
