# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import pytest
import torch
from torchvision import tv_tensors

from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import SampleBatch


@pytest.fixture
def fxt_keypoint_det_batch_data_entity() -> SampleBatch:
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
    keypoints_visible = torch.randint(0, 1, (batch_size, 17))
    keypoints = torch.cat([keypoints, keypoints_visible.unsqueeze(-1)], dim=-1)
    labels = torch.ones(batch_size, dtype=torch.long)

    return SampleBatch(
        images=tv_tensor,
        imgs_info=img_infos,
        bboxes=[bboxes for _ in range(batch_size)],
        labels=list(labels),
        keypoints=list(keypoints),
    )
