# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from omegaconf import DictConfig
from otx.core.data.dataset.classification import MulticlassClsBatchDataEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import HlabelClsBatchDataEntity, MultilabelClsBatchDataEntity
from otx.core.types.label import HLabelInfo
from torchvision import tv_tensors


@pytest.fixture()
def fxt_hlabel_data() -> HLabelInfo:
    return HLabelInfo(
        label_names=[
            "Heart",
            "Spade",
            "Heart_Queen",
            "Heart_King",
            "Spade_A",
            "Spade_King",
        ],
        label_groups=[
            ["Heart", "Spade"],
            ["Heart_Queen", "Heart_King"],
            ["Spade_A", "Spade_King"],
        ],
        num_multiclass_heads=3,
        num_multilabel_classes=0,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4), "2": (4, 6)},
        num_single_label_classes=0,
        empty_multiclass_head_indices=[],
        class_to_group_idx={
            "Heart": (0, 0),
            "Spade": (0, 1),
            "Heart_Queen": (1, 0),
            "Heart_King": (1, 1),
            "Spade_A": (2, 0),
            "Spade_King": (2, 1),
        },
        all_groups=[
            ["Heart", "Spade"],
            ["Heart_Queen", "Heart_King"],
            ["Spade_A", "Spade_King"],
        ],
        label_to_idx={
            "Heart": 0,
            "Spade": 1,
            "Heart_Queen": 2,
            "Heart_King": 3,
            "Spade_A": 4,
            "Spade_King": 5,
        },
        label_tree_edges=[
            ["Heart_Queen", "Heart"],
            ["Heart_King", "Heart"],
            ["Spade_A", "Spade"],
            ["Spade_King", "Spade"],
        ],
    )


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
        num_single_label_classes=6,
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


@pytest.fixture()
def fxt_hlabel_cifar() -> HLabelInfo:
    return HLabelInfo(
        label_names=[
            "beaver",
            "dolphin",
            "otter",
            "seal",
            "whale",
            "aquarium_fish",
            "flatfish",
            "ray",
            "shark",
            "trout",
            "aquatic_mammals",
            "fish",
        ],
        label_groups=[
            ["beaver", "dolphin", "otter", "seal", "whale"],
            ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            ["aquatic_mammals", "fish"],
        ],
        num_multiclass_heads=3,
        num_multilabel_classes=0,
        head_idx_to_logits_range={"0": (0, 5), "1": (5, 10), "2": (10, 12)},
        num_single_label_classes=12,
        empty_multiclass_head_indices=[],
        class_to_group_idx={
            "beaver": (0, 0),
            "dolphin": (0, 1),
            "otter": (0, 2),
            "seal": (0, 3),
            "whale": (0, 4),
            "aquarium_fish": (1, 0),
            "flatfish": (1, 1),
            "ray": (1, 2),
            "shark": (1, 3),
            "trout": (1, 4),
            "aquatic_mammals": (2, 0),
            "fish": (2, 1),
        },
        all_groups=[
            ["beaver", "dolphin", "otter", "seal", "whale"],
            ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            ["aquatic_mammals", "fish"],
        ],
        label_to_idx={
            "aquarium_fish": 0,
            "beaver": 1,
            "dolphin": 2,
            "flatfish": 3,
            "otter": 4,
            "ray": 5,
            "seal": 6,
            "shark": 7,
            "trout": 8,
            "whale": 9,
            "aquatic_mammals": 10,
            "fish": 11,
        },
        label_tree_edges=[
            ["aquarium_fish", "fish"],
            ["beaver", "aquatic_mammals"],
            ["dolphin", "aquatic_mammals"],
            ["otter", "aquatic_mammals"],
            ["seal", "aquatic_mammals"],
            ["whale", "aquatic_mammals"],
            ["flatfish", "aquarium_fish"],
            ["ray", "aquarium_fish"],
            ["shark", "aquarium_fish"],
            ["trout", "aquarium_fish"],
        ],
    )


@pytest.fixture()
def fxt_multiclass_cls_batch_data_entity() -> MulticlassClsBatchDataEntity:
    batch_size = 2
    random_tensor = torch.randn((batch_size, 3, 224, 224))
    tv_tensor = tv_tensors.Image(data=random_tensor)
    img_infos = [ImageInfo(img_idx=i, img_shape=(224, 224), ori_shape=(224, 224)) for i in range(batch_size)]
    return MulticlassClsBatchDataEntity(
        batch_size=2,
        images=tv_tensor,
        imgs_info=img_infos,
        labels=[torch.tensor([0]), torch.tensor([1])],
    )


@pytest.fixture()
def fxt_multilabel_cls_batch_data_entity(
    fxt_multiclass_cls_batch_data_entity,
    fxt_multilabel_labelinfo,
) -> MultilabelClsBatchDataEntity:
    return MultilabelClsBatchDataEntity(
        batch_size=2,
        images=fxt_multiclass_cls_batch_data_entity.images,
        imgs_info=fxt_multiclass_cls_batch_data_entity.imgs_info,
        labels=[
            torch.nn.functional.one_hot(label, num_classes=fxt_multilabel_labelinfo.num_classes).flatten()
            for label in fxt_multiclass_cls_batch_data_entity.labels
        ],
    )


@pytest.fixture()
def fxt_hlabel_cls_batch_data_entity(fxt_multilabel_cls_batch_data_entity) -> HlabelClsBatchDataEntity:
    return HlabelClsBatchDataEntity(**asdict(fxt_multilabel_cls_batch_data_entity))


@pytest.fixture()
def fxt_config_mock() -> DictConfig:
    pseudo_model_config = {
        "backbone": {
            "name": "dinov2_vits14_reg",
            "frozen": False,
        },
        "head": {
            "in_channels": 384,
            "num_classes": 2,
        },
        "data_preprocess": {
            "mean": [1, 1, 1],
            "std": [1, 1, 1],
            "to_rgb": True,
        },
    }
    return DictConfig(pseudo_model_config)
