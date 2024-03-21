# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity, SegDataEntity
from otx.core.data.mem_cache import MemCacheHandlerSingleton
from torchvision.tv_tensors import Image, Mask


@pytest.fixture(scope="session")
def fxt_seg_data_entity() -> tuple[tuple, SegDataEntity, SegBatchDataEntity]:
    img_size = (32, 32)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_masks = Mask(torch.randint(low=0, high=255, size=img_size, dtype=torch.uint8))
    # define data entity
    single_data_entity = SegDataEntity(fake_image, fake_image_info, fake_masks)
    batch_data_entity = SegBatchDataEntity(
        1,
        [Image(data=torch.from_numpy(fake_image))],
        [fake_image_info],
        [fake_masks],
    )
    batch_pred_data_entity = SegBatchPredEntity(
        1,
        [Image(data=torch.from_numpy(fake_image))],
        [fake_image_info],
        [],
        [fake_masks],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(autouse=True)
def fxt_clean_up_mem_cache() -> None:
    """Clean up the mem-cache instance at the end of the test.

    It is required for everyone who tests model training pipeline.
    See https://github.com/openvinotoolkit/training_extensions/actions/runs/7326689283/job/19952721142?pr=2749#step:5:3098
    """
    yield
    MemCacheHandlerSingleton.delete()


# TODO(Jaeguk): Add cpu param when OTX can run integration test parallelly for each task.
@pytest.fixture(params=[pytest.param("gpu", marks=pytest.mark.gpu)])
def fxt_accelerator(request: pytest.FixtureRequest) -> str:
    return request.param
