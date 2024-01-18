# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import pytest
import torch
from mmpretrain.structures import DataSample
from omegaconf import DictConfig
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import HLabelInfo, MulticlassClsBatchDataEntity
from torchvision import tv_tensors


@pytest.fixture()
def fxt_data_sample() -> DataSample:
    data_sample = DataSample(
        img_shape=(24, 24, 3),
        gt_label=torch.zeros(6, dtype=torch.long),
    )
    return [data_sample, data_sample]


@pytest.fixture()
def fxt_hlabel_info() -> HLabelInfo:
    return HLabelInfo(
        num_multiclass_heads=3,
        num_multilabel_classes=0,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4), "2": (4, 6)},
        num_single_label_classes=6,
        empty_multiclass_head_indices=[],
        class_to_group_idx={
            "0": (0, 0),
            "1": (0, 1),
            "2": (1, 0),
            "3": (1, 1),
            "4": (2, 0),
            "5": (2, 1),
        },
        all_groups=[["0", "1"], ["2", "3"], ["4", "5"]],
        label_to_idx={
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
        },
        label_tree_edges=[
            ["0", "0"], ["1", "0"], ["2", "1"], ["3", "1"], ["4", "2"], ["5", "2"]
        ]
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
