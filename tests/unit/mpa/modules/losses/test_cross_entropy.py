# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.losses.cross_entropy_loss import (
    CrossEntropyLossWithIgnore,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCrossEntropyLossWithIgnore:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.num_classes = 2
        self.default_gt = torch.zeros(2).long()
        self.default_input = torch.zeros((2, self.num_classes))
        self.default_input[0, 0] = 1
        self.default_loss = CrossEntropyLossWithIgnore()

    @e2e_pytest_unit
    def test_forward(self):
        result_c = self.default_loss(self.default_input, self.default_gt)
        self.default_input[0, 1] = 1
        result_w = self.default_loss(self.default_input, self.default_gt)
        assert result_c < result_w

    def test_weight(self):
        result = self.default_loss(self.default_input, self.default_gt)
        loss_w = CrossEntropyLossWithIgnore(loss_weight=2)
        result_w = loss_w(self.default_input, self.default_gt)
        assert result_w > result

    def test_ignore(self):
        loss_i = CrossEntropyLossWithIgnore(ignore_index=0)
        result_i = loss_i(self.default_input, self.default_gt)
        assert result_i == 0

    def test_reduction(self):
        result = self.default_loss(self.default_input, self.default_gt)
        loss_s = CrossEntropyLossWithIgnore(reduction="sum")
        result_s = loss_s(self.default_input, self.default_gt)
        assert result_s > result
