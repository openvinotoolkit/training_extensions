# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.dataset_base import DatasetItem
from otx.algo.samplers.class_incremental_sampler import ClassIncrementalSampler
from otx.core.data.dataset.base import OTXDataset
from otx.core.utils.utils import get_idx_list_per_classes


@pytest.fixture()
def fxt_old_new_dataset() -> OTXDataset:
    dataset_items = (
        [
            DatasetItem(
                id=f"item00{i}_0",
                subset="train",
                media=None,
                annotations=[
                    Label(label=0),
                ],
            )
            for i in range(1, 101)
        ]
        + [
            DatasetItem(
                id=f"item00{i}_1",
                subset="train",
                media=None,
                annotations=[
                    Label(label=1),
                ],
            )
            for i in range(1, 9)
        ]
        + [
            DatasetItem(
                id=f"item00{i}_2",
                subset="train",
                media=None,
                annotations=[
                    Label(label=2),
                ],
            )
            for i in range(1, 9)
        ]
    )

    dm_dataset = DmDataset.from_iterable(dataset_items, categories=["0", "1", "2"])
    return OTXDataset(
        dm_subset=dm_dataset.get_subset("train"),
        transforms=[],
    )


class TestBalancedSampler:
    def test_sampler_iter(self, fxt_old_new_dataset):
        sampler = ClassIncrementalSampler(
            fxt_old_new_dataset,
            batch_size=4,
            old_classes=["0", "1"],
            new_classes=["2"],
        )
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)
        assert len(sampler.old_indices) == 108  # "0" + "1"
        assert len(sampler.new_indices) == 8  # "2"
        assert sampler.old_new_ratio == 3  # np.sqrt(108 / 8)
        assert sampler.num_samples == len(fxt_old_new_dataset)

    def test_sampler_efficient_mode(self, fxt_old_new_dataset):
        sampler = ClassIncrementalSampler(
            fxt_old_new_dataset,
            batch_size=4,
            old_classes=["0", "1"],
            new_classes=["2"],
            efficient_mode=True,
        )
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)
        assert len(sampler.old_indices) == 108  # "0" + "1"
        assert len(sampler.new_indices) == 8  # "2"
        assert sampler.old_new_ratio == 1  # efficient_mode
        assert sampler.data_length == 37  # 37

    def test_sampler_iter_per_class(self, fxt_old_new_dataset):
        batch_size = 4
        sampler = ClassIncrementalSampler(
            fxt_old_new_dataset,
            batch_size=batch_size,
            old_classes=["0", "1"],
            new_classes=["2"],
        )

        stats = get_idx_list_per_classes(fxt_old_new_dataset.dm_subset, True)
        old_idx = stats["0"] + stats["1"]
        new_idx = stats["2"]
        list_iter = list(iter(sampler))
        for i in range(0, len(sampler), batch_size):
            batch = sorted(list_iter[i : i + batch_size])
            assert all(idx in old_idx for idx in batch[: sampler.old_new_ratio])
            assert all(idx in new_idx for idx in batch[sampler.old_new_ratio :])
