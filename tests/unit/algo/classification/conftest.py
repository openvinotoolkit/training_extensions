# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from unittest.mock import MagicMock

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
    img_infos = [ImageInfo(
        img_idx=i,
        img_shape=(224, 224),
        ori_shape=(224, 224),
        pad_shape=(0, 0),
        scale_factor=(1.0, 1.0),
    ) for i in range(batch_size)]
    return MulticlassClsBatchDataEntity(
        batch_size=2,
        images=tv_tensor,
        imgs_info=img_infos,
        labels=[torch.tensor([0]), torch.tensor([1])],
    )

@pytest.fixture()
def fxt_config_mock() -> DictConfig:
    config_mock = MagicMock(spec=DictConfig)
    config_mock.backbone = MagicMock(spec=DictConfig)
    config_mock.backbone.name = "dinov2_vits14_reg"
    config_mock.backbone.frozen = False

    config_mock.head = MagicMock(spec=DictConfig)
    config_mock.head.in_channels = 384
    config_mock.head.num_classes = 2

    config_mock.data_preprocess = MagicMock(spec=DictConfig)
    config_mock.data_preprocess.to_rgb = True
    config_mock.data_preprocess.mean = [1, 1, 1]
    config_mock.data_preprocess.std = [1, 1, 1]
    return config_mock
