# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests of incremental sampler."""

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
from getitune.data.samplers.class_incremental_sampler import ClassIncrementalSampler


@pytest.fixture
def fxt_old_new_dataset() -> VisionDataset:
    categories = {"label": LabelCategories(labels=("0", "1", "2"))}
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
    for _ in range(1, 9):
        dm_dataset.append(
            ClassificationSample(
                image=tv_tensors.Image(torch.zeros(3, 10, 10, dtype=torch.uint8)),
                label=torch.tensor(2, dtype=torch.uint16),
                dm_image_info=DmImageInfo(width=10, height=10),
                subset=Subset.TRAINING,
            ),
        )

    return MulticlassClsDataset(
        dm_subset=dm_dataset,
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

        stats = fxt_old_new_dataset.get_idx_list_per_classes(True)
        old_idx = stats["0"] + stats["1"]
        new_idx = stats["2"]
        list_iter = list(iter(sampler))
        for i in range(0, len(sampler), batch_size):
            batch = sorted(list_iter[i : i + batch_size])
            assert all(idx in old_idx for idx in batch[: sampler.old_new_ratio])
            assert all(idx in new_idx for idx in batch[sampler.old_new_ratio :])
