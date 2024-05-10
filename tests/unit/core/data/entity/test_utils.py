# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of utils for data entity."""

import torch
from otx.core.data.entity.base import ImageInfo, OTXBatchDataEntity
from otx.core.data.entity.utils import stack_batch


def test_stack_batch():
    # Create a sample entity with tensor images
    entity = OTXBatchDataEntity(
        batch_size=3,
        images=[
            torch.tensor([[[1, 2], [3, 4], [5, 6]]]),
            torch.tensor([[[5, 6, 7], [8, 9, 10]]]),
            torch.tensor([[[11, 12, 13, 14], [15, 16, 17, 18]]]),
            torch.tensor([[[19, 20, 0], [0, 0, 0]]]),
        ],
        imgs_info=[
            ImageInfo(img_shape=(3, 2), img_idx=0, ori_shape=(2, 2)),
            ImageInfo(img_shape=(2, 3), img_idx=1, ori_shape=(2, 3)),
            ImageInfo(img_shape=(2, 4), img_idx=2, ori_shape=(2, 4)),
            ImageInfo(img_shape=(2, 3), img_idx=3, ori_shape=(1, 2), padding=(0, 0, 1, 1)),  # previously padded image
        ],
    )

    # Call the stack_batch function
    stacked_images, batch_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=1, pad_value=0)

    # Assert the output
    assert len(stacked_images) == 4
    assert stacked_images[0].tolist() == [[[1, 2, 0, 0], [3, 4, 0, 0], [5, 6, 0, 0]]]
    assert stacked_images[1].tolist() == [[[5, 6, 7, 0], [8, 9, 10, 0], [0, 0, 0, 0]]]
    assert stacked_images[2].tolist() == [[[11, 12, 13, 14], [15, 16, 17, 18], [0, 0, 0, 0]]]
    assert stacked_images[3].tolist() == [[[19, 20, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

    assert batch_info[0].img_shape == (3, 4)
    assert batch_info[1].img_shape == (3, 4)
    assert batch_info[2].img_shape == (3, 4)
    assert batch_info[3].img_shape == (3, 4)

    assert batch_info[0].ori_shape == (2, 2)
    assert batch_info[1].ori_shape == (2, 3)
    assert batch_info[2].ori_shape == (2, 4)
    assert batch_info[3].ori_shape == (1, 2)

    assert batch_info[0].padding == (0, 0, 2, 0)
    assert batch_info[1].padding == (0, 0, 1, 1)
    assert batch_info[2].padding == (0, 0, 0, 1)
    assert batch_info[3].padding == (0, 0, 2, 2)
