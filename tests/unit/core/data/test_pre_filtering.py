# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from datumaro.components.annotation import AnnotationType, Bbox, Label, Polygon
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
                Bbox(x=0, y=0, w=1, h=1, label=0),
                Label(label=i % 3),
            ],
        )
        for i in range(1, 81)
    ]
    dataset_items.append(
        DatasetItem(
            id="item000_wrong_bbox",
            subset="train",
            media=None,
            annotations=[
                Bbox(x=0, y=0, w=-1, h=-1, label=0),
                Label(label=0),
            ],
        ),
    )
    dataset_items.append(
        DatasetItem(
            id="item000_wrong_polygon",
            subset="train",
            media=None,
            annotations=[
                Bbox(x=0, y=0, w=-1, h=-1, label=0),
                Polygon(points=[0.1, 0.2, 0.1, 0.2, 0.1, 0.2], label=0),
                Label(label=0),
            ],
        ),
    )
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
    return DmDataset.from_iterable(dataset_items, categories=["0", "1", "2", "3"])


@pytest.mark.parametrize("unannotated_items_ratio", [0.0, 0.1, 0.5, 1.0])
def test_pre_filtering(fxt_dm_dataset_with_unannotated: DmDataset, unannotated_items_ratio: float) -> None:
    """Test function for pre_filtering.

    Args:
        fxt_dm_dataset_with_unannotated (DmDataset): The dataset to be filtered.
        unannotated_items_ratio (float): The ratio of unannotated background items to be added.

    Returns:
        None
    """
    empty_items = [
        item for item in fxt_dm_dataset_with_unannotated if item.subset == "train" and len(item.annotations) == 0
    ]
    assert len(fxt_dm_dataset_with_unannotated) == 102
    assert len(empty_items) == 20

    filtered_dataset = pre_filtering(
        dataset=fxt_dm_dataset_with_unannotated,
        data_format="datumaro",
        unannotated_items_ratio=unannotated_items_ratio,
    )
    assert len(filtered_dataset) == 82 + int(len(empty_items) * unannotated_items_ratio)
    assert len(filtered_dataset.categories()[AnnotationType.label]) == 3
