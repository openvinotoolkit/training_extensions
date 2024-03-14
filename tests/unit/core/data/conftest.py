# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from datumaro.components.annotation import Bbox, Label, Mask
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
from otx.core.data.dataset.segmentation import (
    OTXSegmentationDataset,
    SegDataEntity,
)
from otx.core.data.mem_cache import MemCacheHandlerSingleton

if TYPE_CHECKING:
    from otx.core.data.dataset.base import OTXDataset, T_OTXDataEntity
    from otx.core.data.mem_cache import MemCacheHandlerBase
    from pytest_mock import MockerFixture


@pytest.fixture()
def fxt_mem_cache_handler(monkeypatch) -> MemCacheHandlerBase:
    monkeypatch.setattr(MemCacheHandlerSingleton, "check_system_memory", lambda *_: True)
    handler = MemCacheHandlerSingleton.create(mode="singleprocessing", mem_size=1024 * 1024)
    yield handler
    MemCacheHandlerSingleton.delete()


@pytest.fixture(params=["bytes", "numpy"])
def fxt_dm_item(request) -> DatasetItem:
    np_img = np.zeros(shape=(10, 10, 3), dtype=np.uint8)
    np_img[:, :, 0] = 0  # Set 0 for B channel
    np_img[:, :, 1] = 1  # Set 1 for G channel
    np_img[:, :, 2] = 2  # Set 2 for R channel

    if request.param == "bytes":
        _, np_bytes = cv2.imencode(".png", np_img)
        media = Image.from_bytes(np_bytes.tobytes())
    elif request.param == "numpy":
        media = Image.from_numpy(np_img)
    else:
        raise ValueError(request.param)

    return DatasetItem(
        id="item",
        subset="train",
        media=media,
        annotations=[
            Label(label=0),
            Bbox(x=0, y=0, w=1, h=1, label=0),
            Mask(label=0, image=np.zeros(shape=(10, 10), dtype=np.uint8)),
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
        (OTXSegmentationDataset, SegDataEntity),
    ],
    ids=["multi_class_cls", "detection", "semantic_seg"],
)
def fxt_dataset_and_data_entity_cls(
    request: pytest.FixtureRequest,
) -> tuple[OTXDataset, T_OTXDataEntity]:
    return request.param
