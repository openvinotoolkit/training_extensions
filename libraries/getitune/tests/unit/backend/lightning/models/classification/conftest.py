# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig
from torchvision import tv_tensors

from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import SampleBatch


@pytest.fixture
def fxt_multiclass_cls_batch_data_entity() -> SampleBatch:
    batch_size = 2
    random_tensor = torch.randn((batch_size, 3, 224, 224))
    tv_tensor = tv_tensors.Image(data=random_tensor)
    img_infos = [ImageInfo(img_idx=i, img_shape=(224, 224), ori_shape=(224, 224)) for i in range(batch_size)]
    return SampleBatch(images=tv_tensor, imgs_info=img_infos, labels=[torch.tensor([0]), torch.tensor([1])])


@pytest.fixture
def fxt_multilabel_cls_batch_data_entity(fxt_multiclass_cls_batch_data_entity, fxt_multilabel_labelinfo) -> SampleBatch:
    return SampleBatch(
        images=fxt_multiclass_cls_batch_data_entity.images,
        imgs_info=fxt_multiclass_cls_batch_data_entity.imgs_info,
        labels=[
            torch.nn.functional.one_hot(label, num_classes=fxt_multilabel_labelinfo.num_classes).flatten()
            for label in fxt_multiclass_cls_batch_data_entity.labels
        ],
    )


@pytest.fixture
def fxt_config_mock() -> DictConfig:
    return DictConfig(
        {
            "backbone": {"name": "dinov2_vits14_reg", "frozen": False},
            "head": {"in_channels": 384, "num_classes": 2},
            "data_preprocess": {"mean": [1, 1, 1], "std": [1, 1, 1], "to_rgb": True},
        },
    )
