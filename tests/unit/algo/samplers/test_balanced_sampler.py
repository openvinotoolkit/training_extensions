# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.dataset_base import DatasetItem
from otx.algo.samplers.balanced_sampler import BalancedSampler
from otx.core.data.dataset.base import OTXDataset
from otx.core.utils.utils import get_idx_list_per_classes


@pytest.fixture()
def fxt_imbalanced_dataset() -> OTXDataset:
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
        for i in range(1, 9)
    ]

    dm_dataset = DmDataset.from_iterable(dataset_items, categories=["0", "1"])
    return OTXDataset(
        dm_subset=dm_dataset.get_subset("train"),
        transforms=[],
    )


class TestBalancedSampler:
    def test_sampler_iter(self, fxt_imbalanced_dataset):
        sampler = BalancedSampler(fxt_imbalanced_dataset)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)
        assert sampler.num_trials == math.ceil(len(fxt_imbalanced_dataset) / sampler.num_cls)

    def test_sampler_efficient_mode(self, fxt_imbalanced_dataset):
        sampler = BalancedSampler(fxt_imbalanced_dataset, efficient_mode=True)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)
        assert sampler.num_trials == 51

    def test_sampler_iter_with_multiple_replicas(self, fxt_imbalanced_dataset):
        sampler = BalancedSampler(fxt_imbalanced_dataset, num_replicas=2)
        sampler_iter = iter(sampler)
        count = 0

        for _ in sampler_iter:
            count += 1

        assert count == len(sampler)

    def test_compute_class_statistics(self, fxt_imbalanced_dataset):
        # Compute class statistics
        stats = get_idx_list_per_classes(fxt_imbalanced_dataset.dm_subset)

        # Check the expected results
        assert stats == {0: list(range(100)), 1: list(range(100, 108))}

    def test_sampler_iter_per_class(self, fxt_imbalanced_dataset):
        batch_size = 4
        sampler = BalancedSampler(fxt_imbalanced_dataset)

        stats = get_idx_list_per_classes(fxt_imbalanced_dataset.dm_subset)
        class_0_idx = stats[0]
        class_1_idx = stats[1]
        list_iter = list(iter(sampler))
        for i in range(0, len(sampler), batch_size):
            batch = sorted(list_iter[i : i + batch_size])
            assert all(idx in class_0_idx for idx in batch[:2])
            assert all(idx in class_1_idx for idx in batch[2:])
