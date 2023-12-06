# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import DatasetSubset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from otx.core.data.dataset.classification import (
    MulticlassClsDataEntity,
    OTXMulticlassClsDataset,
)
from otx.core.data.dataset.detection import (
    DetDataEntity,
    OTXDetectionDataset,
)
from otx.core.data.mem_cache import MemCacheHandlerSingleton

if TYPE_CHECKING:
    from otx.core.data.dataset.base import OTXDataset, T_OTXDataEntity
    from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def fxt_mem_cache() -> None:
    MemCacheHandlerSingleton.create(mode="singleprocessing", mem_size=1024 * 1024)
    yield
    MemCacheHandlerSingleton.delete()


@pytest.fixture()
def fxt_dm_item() -> DatasetItem:
    return DatasetItem(
        id="item",
        subset="train",
        media=Image.from_numpy(np.zeros(shape=(10, 10, 3), dtype=np.uint8)),
        annotations=[
            Label(label=0),
            Bbox(x=0, y=0, w=1, h=1, label=0),
        ],
    )


@pytest.fixture()
def fxt_mock_dm_subset(mocker: MockerFixture, fxt_dm_item: DatasetItem) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=DatasetSubset)
    mock_dm_subset.name = fxt_dm_item.subset
    mock_dm_subset.get.return_value = fxt_dm_item
    mock_dm_subset.__iter__.return_value = iter([fxt_dm_item])
    return mock_dm_subset


@pytest.fixture(
    params=[
        (OTXMulticlassClsDataset, MulticlassClsDataEntity),
        (OTXDetectionDataset, DetDataEntity),
    ],
)
def fxt_dataset_and_data_entity_cls(
    request: pytest.FixtureRequest,
) -> tuple[OTXDataset, T_OTXDataEntity]:
    return request.param
