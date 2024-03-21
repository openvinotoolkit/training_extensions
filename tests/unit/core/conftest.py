# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
import torch
from datumaro import Label
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
)
from otx.core.types.label import HLabelInfo, LabelInfo, NullLabelInfo, SegLabelInfo
from torchvision import tv_tensors


@pytest.fixture(scope="session", autouse=True)
def fxt_register_configs() -> None:
    register_configs()


@pytest.fixture(scope="session", autouse=True)
def fxt_null_label_info() -> LabelInfo:
    return NullLabelInfo()


@pytest.fixture(scope="session", autouse=True)
def fxt_seg_label_info() -> SegLabelInfo:
    label_names = ["class1", "class2", "class3"]
    return SegLabelInfo(
        label_names=label_names,
        label_groups=[
            label_names,
            ["class2", "class3"],
        ],
    )


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


@pytest.fixture()
def fxt_hlabel_multilabel_info() -> HLabelInfo:
    return HLabelInfo(
        label_names=[
            "Heart",
            "Spade",
            "Heart_Queen",
            "Heart_King",
            "Spade_A",
            "Spade_King",
            "Black_Joker",
            "Red_Joker",
            "Extra_Joker",
        ],
        label_groups=[
            ["Heart", "Spade"],
            ["Heart_Queen", "Heart_King"],
            ["Spade_A", "Spade_King"],
            ["Black_Joker"],
            ["Red_Joker"],
            ["Extra_Joker"],
        ],
        num_multiclass_heads=3,
        num_multilabel_classes=3,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4), "2": (4, 6)},
        num_single_label_classes=3,
        empty_multiclass_head_indices=[],
        class_to_group_idx={
            "Heart": (0, 0),
            "Spade": (0, 1),
            "Heart_Queen": (1, 0),
            "Heart_King": (1, 1),
            "Spade_A": (2, 0),
            "Spade_King": (2, 1),
            "Black_Joker": (3, 0),
            "Red_Joker": (3, 1),
            "Extra_Joker": (3, 2),
        },
        all_groups=[
            ["Heart", "Spade"],
            ["Heart_Queen", "Heart_King"],
            ["Spade_A", "Spade_King"],
            ["Black_Joker"],
            ["Red_Joker"],
            ["Extra_Joker"],
        ],
        label_to_idx={
            "Heart": 0,
            "Spade": 1,
            "Heart_Queen": 2,
            "Heart_King": 3,
            "Spade_A": 4,
            "Spade_King": 5,
            "Black_Joker": 6,
            "Red_Joker": 7,
            "Extra_Joker": 8,
        },
        label_tree_edges=[
            ["Heart_Queen", "Heart"],
            ["Heart_King", "Heart"],
            ["Spade_A", "Spade"],
            ["Spade_King", "Spade"],
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
