# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity, SegDataEntity


@pytest.fixture(scope="session")
def fxt_seg_data_entity() -> tuple[tuple, SegDataEntity, SegBatchDataEntity]:
    img_size = (224, 224)
    fake_image = torch.rand(img_size)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_masks = torch.rand(img_size)
    # define data entity
    single_data_entity = SegDataEntity(fake_image, fake_image_info, fake_masks)
    batch_data_entity = SegBatchDataEntity(1, [fake_image], [fake_image_info], [fake_masks])
    batch_pred_data_entity = SegBatchPredEntity(1, [fake_image], [fake_image_info], [], [fake_masks])

    return single_data_entity, batch_pred_data_entity, batch_data_entity
