# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests of balanced sampler."""

import math

import pytest
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torchvision import tv_tensors

from getitune.data.dataset import MulticlassClsDataset
from getitune.data.dataset.base import VisionDataset
from getitune.data.entity.sample import ClassificationSample
from getitune.data.samplers.balanced_sampler import BalancedSampler


@pytest.fixture
def fxt_imbalanced_dataset() -> VisionDataset:
    categories = {"label": LabelCategories(labels=("0", "1"))}
    dm_dataset = Dataset(ClassificationSample, categories=categories)  # type: ignore[arg-type]

    for _ in range(1, 101):
        dm_dataset.append(
            ClassificationSample(
                image=tv_tensors.Image(torch.zeros(3, 10, 10, dtype=torch.uint8)),
                label=torch.tensor(0, dtype=torch.uint16),
                dm_image_info=DmImageInfo(width=10, height=10),
                subset=Subset.TRAINING,
            ),
        )
    for _ in range(1, 9):
        dm_dataset.append(
            ClassificationSample(
                image=tv_tensors.Image(torch.zeros(3, 10, 10, dtype=torch.uint8)),
                label=torch.tensor(1, dtype=torch.uint16),
                dm_image_info=DmImageInfo(width=10, height=10),
                subset=Subset.TRAINING,
            ),
        )

    return MulticlassClsDataset(dm_subset=dm_dataset, transforms=[])


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
        stats = fxt_imbalanced_dataset.get_idx_list_per_classes()

        # Check the expected results
        assert stats == {0: list(range(100)), 1: list(range(100, 108))}

    def test_sampler_iter_per_class(self, fxt_imbalanced_dataset):
        batch_size = 4
        sampler = BalancedSampler(fxt_imbalanced_dataset)

        stats = fxt_imbalanced_dataset.get_idx_list_per_classes()
        class_0_idx = stats[0]
        class_1_idx = stats[1]
        list_iter = list(iter(sampler))
        for i in range(0, len(sampler), batch_size):
            batch = sorted(list_iter[i : i + batch_size])
            assert all(idx in class_0_idx for idx in batch[:2])
            assert all(idx in class_1_idx for idx in batch[2:])
