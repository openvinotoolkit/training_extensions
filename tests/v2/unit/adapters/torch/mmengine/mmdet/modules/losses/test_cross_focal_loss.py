"""Unit tests of otx/v2/adapters/torch/mmengine/mmdet/modules/losses/cross_focal_loss.py"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch

from otx.v2.adapters.torch.mmengine.mmdet.modules.losses import (
    CrossSigmoidFocalLoss,
    OrdinaryFocalLoss,
)


def test_cross_sigmoid_focal_loss():
    criterian = CrossSigmoidFocalLoss(num_classes=5)
    preds = torch.randn(1000, 3)
    targets = torch.randint(0, 3, (1000,))
    weight = torch.randn(1000)
    avg_factor = 85.0
    valid_label_mask = torch.randint(0, 2, (1000, 3))

    out = criterian(preds, targets, weight, avg_factor=avg_factor, valid_label_mask=valid_label_mask)
    assert out.shape == torch.Size([])


def test_ordinary_focal_loss():
    criterian = OrdinaryFocalLoss()
    preds = torch.randn(1000, 3)
    targets = torch.randint(0, 3, (1000,))
    weight = torch.randn(1000)
    avg_factor = 85.0

    out = criterian(preds, targets, weight, avg_factor)
    assert out.shape == torch.Size([])
