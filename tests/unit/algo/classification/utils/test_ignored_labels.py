# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from otx.algo.classification.utils.ignored_labels import get_valid_label_mask
from otx.core.data.entity.base import ImageInfo


def test_get_valid_label_mask():
    img_metas = [
        ImageInfo(ignored_labels=[2, 4], img_idx=0, img_shape=(32, 32), ori_shape=(32, 32)),
        ImageInfo(ignored_labels=[1, 3, 5], img_idx=1, img_shape=(32, 32), ori_shape=(32, 32)),
        ImageInfo(ignored_labels=[0], img_idx=2, img_shape=(32, 32), ori_shape=(32, 32)),
    ]
    num_classes = 6

    expected_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    )

    mask = get_valid_label_mask(img_metas, num_classes)
    assert torch.equal(mask, expected_mask)
