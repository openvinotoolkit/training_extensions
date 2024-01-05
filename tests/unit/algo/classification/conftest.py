# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig
from torchvision import tv_tensors

from src.otx.core.data.entity.base import ImageInfo
from src.otx.core.data.entity.classification import MulticlassClsBatchDataEntity


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
