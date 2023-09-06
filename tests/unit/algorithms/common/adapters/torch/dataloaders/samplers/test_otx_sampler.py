import pytest
import math
from torch.utils.data import Dataset

from otx.algorithms.common.adapters.torch.dataloaders.samplers import OTXSampler
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOTXSampler:
    @pytest.fixture(autouse=True)
    def setup(self):
        class MockDataset(Dataset):
            def __init__(self):
                self.img_indices = {"old": list(range(0, 6)), "new": list(range(6, 10))}

            def __len__(self):
                return 10

        self.mock_dataset = MockDataset()

    @e2e_pytest_unit
    @pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("use_adaptive_repeat", [True, False])
    def test_sampler_iter(self, batch, use_adaptive_repeat):
        sampler = OTXSampler(self.mock_dataset, batch, use_adaptive_repeats=use_adaptive_repeat)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        repeated_len = len(self.mock_dataset) * sampler.repeat
        assert count == repeated_len
