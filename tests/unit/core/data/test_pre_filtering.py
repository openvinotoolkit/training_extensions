# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.dataset_base import DatasetItem
from otx.core.data.pre_filtering import pre_filtering


@pytest.fixture()
def fxt_dm_dataset_with_unannotated() -> DmDataset:
    dataset_items = [
        DatasetItem(
            id=f"item00{i}_non_empty",
            subset="train",
            media=None,
            annotations=[
                Label(label=3 % i),
            ],
        )
        for i in range(1, 81)
    ]
    dataset_items.extend(
        [
            DatasetItem(
                id=f"item00{i}_empty",
                subset="train",
                media=None,
                annotations=[],
            )
            for i in range(20)
        ],
    )
    return DmDataset.from_iterable(dataset_items, categories=["0", "1", "2"])


@pytest.mark.parametrize("unannotated_bg_ratio", [0.0, 0.1, 0.5, 1.0])
def test_pre_filtering(fxt_dm_dataset_with_unannotated: DmDataset, unannotated_bg_ratio: float) -> None:
    """Test function for pre_filtering.

    Args:
        fxt_dm_dataset_with_unannotated (DmDataset): The dataset to be filtered.
        unannotated_bg_ratio (float): The ratio of unannotated background items to be added.

    Returns:
        None
    """
    empty_items = [item for item in fxt_dm_dataset_with_unannotated if len(item.annotations) == 0]
    assert len(fxt_dm_dataset_with_unannotated) == 100
    assert len(empty_items) == 20

    filtered_dataset = pre_filtering(
        dataset=fxt_dm_dataset_with_unannotated,
        data_format="datumaro",
        unannotated_bg_ratio=unannotated_bg_ratio,
    )
    assert len(filtered_dataset) == 80 + int(len(empty_items) * unannotated_bg_ratio)
