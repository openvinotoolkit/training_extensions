# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import pytest
import torch
from torchvision import tv_tensors
from src.otx.core.data.entity.classification import MulticlassClsBatchDataEntity
from src.otx.core.data.entity.base import ImageInfo

@pytest.fixture()
def fxt_multiclass_cls_batch_data_entity() -> MulticlassClsBatchDataEntity:
    batch_size = 2
    random_tensor = torch.randn((batch_size, 3, 32, 32))
    tv_tensor = tv_tensors.Image(data=random_tensor)
    img_infos = [ImageInfo(
        img_idx=i,
        img_shape=(224, 224),
        ori_shape=(224, 224),
        pad_shape=(0, 0),
        scale_factor=(1.0, 1.0)
    ) for i in range(batch_size)]
    return MulticlassClsBatchDataEntity(
        batch_size=2,
        images=tv_tensor,
        imgs_info=img_infos,
        labels=[torch.tensor(0), torch.tensor(1)]
    )