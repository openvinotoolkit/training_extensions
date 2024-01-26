# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from otx.core.data.entity.base import ImageInfo, Points
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    VisualPromptingDataEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingDataEntity,
)
from torchvision import tv_tensors


@pytest.fixture(scope="session")
def fxt_vpm_data_entity() -> (
    tuple[VisualPromptingDataEntity, VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity]
):
    img_size = (1024, 1024)
    fake_image = tv_tensors.Image(torch.rand(img_size))
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(
        [[0, 0, 1, 1]],
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=img_size,
        dtype=torch.float32,
    )
    fake_points = Points([[2, 2]], canvas_size=img_size, dtype=torch.float32)
    fake_masks = tv_tensors.Mask(torch.rand(img_size))
    fake_labels = torch.as_tensor([1], dtype=torch.int64)
    fake_polygons = [None]
    # define data entity
    single_data_entity = VisualPromptingDataEntity(
        image=fake_image,
        img_info=fake_image_info,
        masks=fake_masks,
        labels=fake_labels,
        polygons=fake_polygons,
        bboxes=fake_bboxes,
        points=fake_points,
    )
    batch_data_entity = VisualPromptingBatchDataEntity(
        batch_size=1,
        images=[fake_image],
        imgs_info=[fake_image_info],
        masks=[fake_masks],
        labels=[fake_labels],
        polygons=[fake_polygons],
        bboxes=[fake_bboxes],
        points=[fake_points],
    )
    batch_pred_data_entity = VisualPromptingBatchPredEntity(
        batch_size=1,
        images=[fake_image],
        imgs_info=[fake_image_info],
        masks=[fake_masks],
        labels=[fake_labels],
        polygons=[fake_polygons],
        bboxes=[fake_bboxes],
        points=[fake_points],
        scores=[],
    )

    return single_data_entity, batch_data_entity, batch_pred_data_entity


@pytest.fixture(scope="session")
def fxt_zero_shot_vpm_data_entity() -> (
    tuple[
        ZeroShotVisualPromptingDataEntity,
        ZeroShotVisualPromptingBatchDataEntity,
        ZeroShotVisualPromptingBatchPredEntity,
    ]
):
    img_size = (1024, 1024)
    fake_image = tv_tensors.Image(torch.rand(img_size))
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(
        [[0, 0, 1, 1]],
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=img_size,
        dtype=torch.float32,
    )
    fake_points = Points([[2, 2]], canvas_size=img_size, dtype=torch.float32)
    fake_masks = tv_tensors.Mask(torch.rand(img_size))
    fake_labels = torch.as_tensor([1], dtype=torch.int64)
    fake_polygons = [None]
    # define data entity
    single_data_entity = ZeroShotVisualPromptingDataEntity(
        image=fake_image,
        img_info=fake_image_info,
        masks=fake_masks,
        labels=fake_labels,
        polygons=fake_polygons,
        prompts=[fake_bboxes, fake_points],
    )
    batch_data_entity = ZeroShotVisualPromptingBatchDataEntity(
        batch_size=1,
        images=[fake_image],
        imgs_info=[fake_image_info],
        masks=[fake_masks],
        labels=[fake_labels],
        polygons=[fake_polygons],
        prompts=[[fake_bboxes, fake_points]],
    )
    batch_pred_data_entity = ZeroShotVisualPromptingBatchPredEntity(
        batch_size=1,
        images=[fake_image],
        imgs_info=[fake_image_info],
        masks=[fake_masks],
        labels=[fake_labels],
        polygons=[fake_polygons],
        prompts=[[fake_bboxes, fake_points]],
        scores=[],
    )

    return single_data_entity, batch_data_entity, batch_pred_data_entity
