# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from otx.core.config import register_configs
from otx.core.data.dataset.base import LabelInfo
from otx.core.data.entity.base import ImageInfo, Points
from otx.core.data.entity.classification import HLabelData
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    VisualPromptingDataEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingDataEntity,
)
from torchvision import tv_tensors


@pytest.fixture(scope="session", autouse=True)
def fxt_register_configs() -> None:
    register_configs()


@pytest.fixture(scope="session", autouse=True)
def fxt_multiclass_labelinfo() -> LabelInfo:
    label_names = ["class1", "class2", "class3"]
    return LabelInfo(
        label_names=label_names,
        label_groups=[
            label_names,
            ["class2", "class3"],
        ],
    )


@pytest.fixture(scope="session", autouse=True)
def fxt_multilabel_labelinfo() -> LabelInfo:
    label_names = ["class1", "class2", "class3"]
    return LabelInfo(
        label_names=label_names,
        label_groups=[
            [label_names[0]],
            [label_names[1]],
            [label_names[2]],
        ],
    )


@pytest.fixture()
def fxt_hlabel_multilabel_info() -> HLabelData:
    return HLabelData(
        num_multiclass_heads=3,
        num_multilabel_classes=3,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4), "2": (4, 6)},
        num_single_label_classes=3,
        empty_multiclass_head_indices=[],
        class_to_group_idx={
            "0": (0, 0),
            "1": (0, 1),
            "2": (1, 0),
            "3": (1, 1),
            "4": (2, 0),
            "5": (2, 1),
            "6": (3, 0),
            "7": (3, 1),
            "8": (3, 2),
        },
        all_groups=[["0", "1"], ["2", "3"], ["4", "5"], ["6"], ["7"], ["8"]],
        label_to_idx={
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
        },
        label_tree_edges=[
            ["0", "0"],
            ["1", "0"],
            ["2", "1"],
            ["3", "1"],
            ["4", "2"],
            ["5", "2"],
        ],
    )


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
    fake_labels = {"bboxes": torch.as_tensor([1], dtype=torch.int64)}
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
