# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.adapters.torch.modules.dataloaders.samplers import OTXSampler
from torch.utils.data import Dataset


class TestOTXSampler:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class MockDataset(Dataset):
            def __init__(self) -> None:
                self.img_indices = {"old": list(range(0, 6)), "new": list(range(6, 10))}

            def __len__(self) -> int:
                return 10

        self.mock_dataset = MockDataset()

    @pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
    def test_sampler_iter(self, batch: int) -> None:
        sampler = OTXSampler(self.mock_dataset, batch)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        repeated_len = len(self.mock_dataset) * sampler.repeat
        assert count == repeated_len
