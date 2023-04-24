import pytest
from torch.utils.data import Dataset

from otx.algorithms.common.adapters.torch.dataloaders.samplers import ClsIncrSampler
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestClsIncrSampler:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        class MockDataset(Dataset):
            def __init__(self):
                self.img_indices = {"old": list(range(0, 6)), "new": list(range(6, 10))}

            def __len__(self):
                return 10

        self.mock_dataset = MockDataset()
        mocker.patch(
            "otx.algorithms.common.adapters.torch.dataloaders.samplers.cls_incr_sampler.unwrap_dataset",
            return_value=(self.mock_dataset, 1),
        )

    @e2e_pytest_unit
    def test_sampler_iter(self):
        sampler = ClsIncrSampler(self.mock_dataset, 4)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)

    @e2e_pytest_unit
    def test_sampler_iter_with_multiple_replicas(self):
        sampler = ClsIncrSampler(self.mock_dataset, 4, num_replicas=2)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)
