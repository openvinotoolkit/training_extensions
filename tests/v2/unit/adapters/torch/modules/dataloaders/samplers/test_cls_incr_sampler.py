import math

import pytest
from otx.v2.adapters.torch.modules.dataloaders.samplers import ClsIncrSampler
from torch.utils.data import Dataset


class TestClsIncrSampler:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class MockDataset(Dataset):
            def __init__(self) -> None:
                self.img_indices = {"old": list(range(0, 6)), "new": list(range(6, 10))}

            def __len__(self) -> int:
                return 10

        self.mock_dataset = MockDataset()

    def test_sampler_iter(self) -> None:
        sampler = ClsIncrSampler(self.mock_dataset, 4)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)

    @pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
    def test_sampler_iter_with_adptive_repeat(self, batch: int) -> None:
        sampler = ClsIncrSampler(self.mock_dataset, batch)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        repeated_len = len(self.mock_dataset) * sampler.repeat
        if not sampler.drop_last:
            assert count == math.ceil(repeated_len / batch) * batch
        else:
            assert count == len(self.mock_dataset) * sampler.repeat

    def test_sampler_iter_with_multiple_replicas(self) -> None:
        sampler = ClsIncrSampler(self.mock_dataset, 4, num_replicas=2)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)
