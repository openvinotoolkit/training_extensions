# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import pytest
import torch
from otx.core.data.entity.base import BboxInfo, ImageInfo
from otx.core.data.entity.keypoint_detection import KeypointDetBatchDataEntity
from torchvision import tv_tensors


@pytest.fixture()
def fxt_keypoint_det_batch_data_entity() -> KeypointDetBatchDataEntity:
    batch_size = 2
    random_tensor = torch.randn((batch_size, 3, 192, 256))
    tv_tensor = tv_tensors.Image(data=random_tensor)
    img_infos = [ImageInfo(img_idx=i, img_shape=(192, 256), ori_shape=(192, 256)) for i in range(batch_size)]
    bboxes = tv_tensors.BoundingBoxes(
        [[0, 0, 1, 1], [1, 1, 3, 3]],
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(192, 256),
        dtype=torch.float32,
    )
    keypoints = torch.randn((batch_size, 17, 2))
    keypoints_visible = torch.randn((batch_size, 17))
    labels = torch.ones(batch_size)
    return KeypointDetBatchDataEntity(
        batch_size=2,
        images=tv_tensor,
        imgs_info=img_infos,
        bboxes=bboxes,
        labels=labels,
        bbox_info=BboxInfo(center=(96, 128), scale=(1, 1), rotation=0),
        keypoints=keypoints,
        keypoints_visible=keypoints_visible,
    )
