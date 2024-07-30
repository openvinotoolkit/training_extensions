# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
import torch
from datumaro import Label, Polygon
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.media import Image
from otx.core.config import register_configs
from otx.core.data.entity.base import ImageInfo, Points
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    VisualPromptingDataEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingDataEntity,
    ZeroShotVisualPromptingLabel,
)
from torchvision import tv_tensors


@pytest.fixture(scope="session", autouse=True)
def fxt_register_configs() -> None:
    register_configs()


@pytest.fixture()
def fxt_hlabel_dataset_subset() -> Dataset:
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=0,
                subset="train",
                media=Image.from_numpy(np.zeros((3, 10, 10))),
                annotations=[
                    Label(
                        label=2,
                        id=0,
                        group=1,
                    ),
                ],
            ),
            DatasetItem(
                id=1,
                subset="train",
                media=Image.from_numpy(np.zeros((3, 10, 10))),
                annotations=[
                    Label(
                        label=4,
                        id=0,
                        group=2,
                    ),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories(
                items=[
                    LabelCategories.Category(name="Heart", parent=""),
                    LabelCategories.Category(name="Spade", parent=""),
                    LabelCategories.Category(name="Heart_Queen", parent="Heart"),
                    LabelCategories.Category(name="Heart_King", parent="Heart"),
                    LabelCategories.Category(name="Spade_A", parent="Spade"),
                    LabelCategories.Category(name="Spade_King", parent="Spade"),
                    LabelCategories.Category(name="Black_Joker", parent=""),
                    LabelCategories.Category(name="Red_Joker", parent=""),
                    LabelCategories.Category(name="Extra_Joker", parent=""),
                ],
                label_groups=[
                    LabelCategories.LabelGroup(name="Card", labels=["Heart", "Spade"]),
                    LabelCategories.LabelGroup(name="Heart Group", labels=["Heart_Queen", "Heart_King"]),
                    LabelCategories.LabelGroup(name="Spade Group", labels=["Spade_Queen", "Spade_King"]),
                ],
            ),
        },
    ).get_subset("train")


@pytest.fixture(scope="session")
def fxt_vpm_data_entity() -> (
    tuple[VisualPromptingDataEntity, VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity]
):
    img_size = (1024, 1024)
    fake_image = tv_tensors.Image(torch.ones(img_size))
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(
        [[0, 0, 1, 1]],
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=img_size,
        dtype=torch.float32,
    )
    fake_points = Points([[2, 2]], canvas_size=img_size, dtype=torch.float32)
    fake_masks = tv_tensors.Mask(torch.ones(1, *img_size))
    fake_labels = {"bboxes": torch.as_tensor([1], dtype=torch.int64), "points": torch.as_tensor([1])}
    fake_polygons = [None]
    fake_scores = torch.tensor([[1.0]])
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
        scores=[fake_scores],
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
    fake_image = tv_tensors.Image(torch.ones(img_size))
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(
        [[0, 0, 1, 1]],
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=img_size,
        dtype=torch.float32,
    )
    fake_points = Points([[2, 2]], canvas_size=img_size, dtype=torch.float32)
    fake_masks = tv_tensors.Mask(torch.ones(1, *img_size))
    fake_labels = ZeroShotVisualPromptingLabel(
        prompts=torch.as_tensor([1, 2], dtype=torch.int64),
        masks=torch.as_tensor([1], dtype=torch.int64),
        polygons=torch.as_tensor([2], dtype=torch.int64),
    )
    fake_polygons = [Polygon(points=[1, 1, 1, 2, 2, 2, 2, 1])]
    fake_scores = torch.tensor([[1.0]])
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
        labels=[fake_labels.prompts],
        polygons=[fake_polygons],
        prompts=[[fake_bboxes, fake_points]],
        scores=[fake_scores],
    )

    return single_data_entity, batch_data_entity, batch_pred_data_entity
