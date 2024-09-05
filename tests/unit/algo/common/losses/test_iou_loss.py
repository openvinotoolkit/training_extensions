# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Test of IoULoss.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/tests/test_models/test_losses/test_loss.py
"""

from __future__ import annotations

import pytest
import torch
from otx.algo.detection.losses import IoULoss


class TestIoULoss:
    def test_iou_type_loss_zeros_weight(self) -> None:
        pred = torch.rand((10, 4))
        target = torch.rand((10, 4))
        weight = torch.zeros(10)

        loss = IoULoss()(pred, target, weight)
        assert loss == 0.0

    def test_loss_with_reduction_override(self) -> None:
        pred = torch.rand((10, 4))
        target = (torch.rand((10, 4)),)
        weight = None

        with pytest.raises(AssertionError):
            # only reduction_override from [None, 'none', 'mean', 'sum'] is not allowed
            IoULoss()(pred, target, weight, reduction_override=True)

    @pytest.mark.parametrize("input_shape", [(10, 4), (0, 4)])
    def test_regression_losses(self, input_shape: tuple[int, int]) -> None:
        pred = torch.rand(input_shape)
        target = torch.rand(input_shape)
        weight = torch.rand(input_shape)

        # Test loss forward
        loss = IoULoss()(pred, target)
        assert isinstance(loss, torch.Tensor)

        # Test loss forward with weight
        loss = IoULoss()(pred, target, weight)
        assert isinstance(loss, torch.Tensor)

        # Test loss forward with reduction_override
        loss = IoULoss()(pred, target, reduction_override="mean")
        assert isinstance(loss, torch.Tensor)

        # Test loss forward with avg_factor
        loss = IoULoss()(pred, target, avg_factor=10)
        assert isinstance(loss, torch.Tensor)

        with pytest.raises(ValueError):  # noqa: PT011
            # loss can evaluate with avg_factor only if reduction is None, 'none' or 'mean'.
            IoULoss()(pred, target, avg_factor=10, reduction_override="sum")

        # Test loss forward with avg_factor and reduction
        for reduction_override in [None, "none", "mean"]:
            loss = IoULoss()(pred, target, avg_factor=10, reduction_override=reduction_override)
            assert isinstance(loss, torch.Tensor)
