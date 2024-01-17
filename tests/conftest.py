# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity, SegDataEntity
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    VisualPromptingDataEntity,
)
from otx.core.data.mem_cache import MemCacheHandlerSingleton
from torchvision import tv_tensors


@pytest.fixture(scope="session")
def fxt_seg_data_entity() -> tuple[tuple, SegDataEntity, SegBatchDataEntity]:
    img_size = (224, 224)
    fake_image = torch.rand(img_size)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_masks = torch.rand(img_size)
    # define data entity
    single_data_entity = SegDataEntity(fake_image, fake_image_info, fake_masks)
    batch_data_entity = SegBatchDataEntity(1, [fake_image], [fake_image_info], [fake_masks])
    batch_pred_data_entity = SegBatchPredEntity(1, [fake_image], [fake_image_info], [], [fake_masks])

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_vpm_data_entity() -> tuple[tuple, VisualPromptingDataEntity, VisualPromptingBatchDataEntity]:
    img_size = (1024, 1024)
    fake_image = torch.rand(img_size)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(
        [[0, 0, 1, 1]],
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=img_size,
        dtype=torch.float32,
    )
    fake_masks = torch.rand(img_size)
    fake_labels = torch.as_tensor([1], dtype=torch.int64)
    fake_polygons = [None]
    # define data entity
    single_data_entity = VisualPromptingDataEntity(
        fake_image,
        fake_image_info,
        fake_bboxes,
        fake_masks,
        fake_labels,
        fake_polygons,
    )
    batch_data_entity = VisualPromptingBatchDataEntity(
        1,
        [fake_image],
        [fake_image_info],
        [fake_bboxes],
        [fake_masks],
        [fake_labels],
        [fake_polygons],
    )
    batch_pred_data_entity = VisualPromptingBatchPredEntity(
        1,
        [fake_image],
        [fake_image_info],
        [],
        [fake_bboxes],
        [fake_masks],
        [fake_labels],
        [fake_polygons],
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
