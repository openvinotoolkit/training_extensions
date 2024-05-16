# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.algo.classification.losses.asymmetric_angular_loss_with_ignore import (
    AsymmetricAngularLossWithIgnore,
    asymmetric_angular_loss_with_ignore,
)


class TestAsymmetricAngularLoss:
    @pytest.fixture()
    def fxt_num_classes(self) -> int:
        return 2

    @pytest.fixture()
    def fxt_gt(self, fxt_num_classes) -> torch.Tensor:
        gt = torch.zeros((2, fxt_num_classes))
        gt[0, 0] = 1
        return gt

    @pytest.fixture()
    def fxt_input(self, fxt_num_classes) -> torch.Tensor:
        inputs = torch.zeros((2, fxt_num_classes))
        inputs[0, 0] = 1
        return inputs

    @pytest.fixture()
    def loss(self) -> AsymmetricAngularLossWithIgnore:
        return AsymmetricAngularLossWithIgnore(reduction="mean")

    def test_forward(self, loss, fxt_input, fxt_gt) -> None:
        result_c = loss(fxt_input, fxt_gt)
        fxt_input[0, 1] = 1
        result_w = loss(fxt_input, fxt_gt)
        assert result_c < result_w

        with pytest.raises(ValueError, match="reduction_override should be"):
            loss(fxt_input, fxt_gt, reduction_override="wrong")

    def test_asymmetric_angular_loss_with_ignore(self):
        pred = torch.tensor([0.5, 0.8, 0.2])
        target = torch.tensor([1, 0, 1])
        loss = asymmetric_angular_loss_with_ignore(pred, target)
        assert torch.isclose(loss, torch.tensor(0.3329), rtol=1e-03, atol=1e-05)

        target = torch.tensor([1, 0])
        with pytest.raises(ValueError, match="pred and target should be in the same shape."):
            asymmetric_angular_loss_with_ignore(pred, target)

        target = torch.tensor([1, 0, 1])
        with pytest.raises(ValueError):  # noqa: PT011
            asymmetric_angular_loss_with_ignore(pred, target, weight=torch.tensor([[1, 1, 1], [1, 1, 1]]))
