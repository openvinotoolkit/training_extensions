# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.losses.asymmetric_angular_loss_with_ignore import (
    AsymmetricAngularLossWithIgnore,
)
from otx.algorithms.classification.adapters.mmcls.models.losses.asymmetric_loss_with_ignore import (
    AsymmetricLossWithIgnore,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestAsymmetricLoss:
    @pytest.fixture(autouse=True)
    def loss_type(self) -> None:
        return AsymmetricLossWithIgnore

    @pytest.fixture(autouse=True)
    def setup(self, loss_type) -> None:
        self.num_classes = 2
        self.default_gt = torch.zeros((2, self.num_classes))
        self.default_input = torch.zeros((2, self.num_classes))
        self.default_input[0, 0] = 1
        self.default_gt[0, 0] = 1
        self.default_loss = loss_type(reduction="mean")

    @e2e_pytest_unit
    def test_forward(self) -> None:
        result_c = self.default_loss(self.default_input, self.default_gt)
        self.default_input[0, 1] = 1
        result_w = self.default_loss(self.default_input, self.default_gt)
        assert result_c < result_w

    @e2e_pytest_unit
    def test_weight(self, loss_type) -> None:
        result = self.default_loss(self.default_input, self.default_gt)
        loss_w = loss_type(loss_weight=2, reduction="mean")
        result_w = loss_w(self.default_input, self.default_gt)
        assert result_w > result

    @e2e_pytest_unit
    def test_reduction(self, loss_type) -> None:
        result = self.default_loss(self.default_input, self.default_gt)
        loss_s = loss_type(reduction="sum")
        result_s = loss_s(self.default_input, self.default_gt)
        assert result_s > result

    @e2e_pytest_unit
    def test_gamma_neg(self, loss_type) -> None:
        result = self.default_loss(self.default_input, self.default_gt)
        loss_s = loss_type(gamma_neg=0.0, reduction="mean")
        result_s = loss_s(self.default_input, self.default_gt)
        assert result_s > result

    @e2e_pytest_unit
    def test_gamma_pos(self, loss_type) -> None:
        result = self.default_loss(self.default_input, self.default_gt)
        loss_s = loss_type(gamma_pos=1.0, reduction="mean")
        result_s = loss_s(self.default_input, self.default_gt)
        assert result_s < result


class TestAsymmetricAngularLoss(TestAsymmetricLoss):
    @pytest.fixture(autouse=True)
    def loss_type(self) -> None:
        return AsymmetricAngularLossWithIgnore
