# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import torch
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
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegDataEntity

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


@pytest.fixture(scope="session")
def fxt_seg_data_entity() -> tuple[tuple, SegDataEntity, SegBatchDataEntity]:
    img_size = (224, 224)
    fake_image = torch.rand(img_size)
    fake_image_info = ImageInfo(0, img_size[0], img_size[0],
                           img_size[0], img_size[0])
    fake_masks = torch.rand(img_size)
    # define data entity
    single_data_entity = SegDataEntity(fake_image,
                                fake_image_info,
                                fake_masks)
    batch_data_entity = SegBatchDataEntity(1, [fake_image],
                                           [fake_image_info],
                                           [fake_masks])
    batch_pred_data_entity = SegBatchPredEntity(1, [fake_image],
                                           [fake_image_info], [],
                                           [fake_masks])

    return single_data_entity, batch_pred_data_entity, batch_data_entity


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
