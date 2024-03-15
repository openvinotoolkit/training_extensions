import pytest
from torch.utils.data import Dataset

from otx.algorithms.common.adapters.torch.dataloaders.samplers import BalancedSampler
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestBalancedSampler:
    @pytest.fixture(autouse=True)
    def setup(self):
        class MockDataset(Dataset):
            def __init__(self):
                self.img_indices = {"foo": list(range(0, 6)), "bar": list(range(6, 10))}

            def __len__(self):
                return 10

        self.mock_dataset = MockDataset()

    @e2e_pytest_unit
    def test_sampler_iter(self):
        sampler = BalancedSampler(self.mock_dataset, 4)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)

    @e2e_pytest_unit
    @pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
    def test_sampler_iter_with_adptive_repeat(self, batch):
        sampler = BalancedSampler(self.mock_dataset, batch)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1
        assert count == len(self.mock_dataset) * sampler.repeat

    @e2e_pytest_unit
    def test_sampler_iter_with_multiple_replicas(self):
        sampler = BalancedSampler(self.mock_dataset, 4, num_replicas=2)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)
