import pytest
from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.dataset_base import DatasetItem
from otx.algo.samplers.repeat_sampler import RepeatSampler, get_proper_repeat_times
from otx.core.data.dataset.base import OTXDataset


@pytest.fixture()
def fxt_otx_dataset() -> OTXDataset:
    dataset_items = [
        DatasetItem(
            id=f"item00{i}_0",
            subset="train",
            media=None,
            annotations=[
                Label(label=0),
            ],
        )
        for i in range(1, 101)
    ] + [
        DatasetItem(
            id=f"item00{i}_1",
            subset="train",
            media=None,
            annotations=[
                Label(label=1),
            ],
        )
        for i in range(1, 51)
    ]

    dm_dataset = DmDataset.from_iterable(dataset_items, categories=["0", "1"])
    return OTXDataset(
        dm_subset=dm_dataset,
        transforms=[],
    )


class TestOTXSampler:
    @pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
    def test_sampler_iter(self, batch, fxt_otx_dataset):
        sampler = RepeatSampler(fxt_otx_dataset, batch)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        repeated_len = len(fxt_otx_dataset) * sampler.repeat
        assert count == repeated_len

        sampler = RepeatSampler(fxt_otx_dataset, batch, n_repeats=3)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        repeated_len = len(fxt_otx_dataset) * sampler.repeat
        assert count == repeated_len

    def test_get_proper_repeat_times(self):
        # Test case 1: data_size = 0, batch_size = 0
        assert get_proper_repeat_times(0, 0, 0.5, 1.0) == 1

        # Test case 2: data_size = 100, batch_size = 10, coef = 0.5, min_repeat = 1.0
        assert get_proper_repeat_times(100, 10, 0.5, 1.0) == 6

        # Test case 3: data_size = 1000, batch_size = 32, coef = 0.8, min_repeat = 2.0
        assert get_proper_repeat_times(1000, 32, 0.8, 2.0) == 9

        # Test case 4: data_size = 500, batch_size = 20, coef = 0.6, min_repeat = 3.0
        assert get_proper_repeat_times(500, 20, 0.6, 3.0) == 7
