# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.heads.custom_cls_head import (
    CustomLinearClsHead,
    CustomNonLinearClsHead,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestLinearClsHead:
    @pytest.fixture(autouse=True)
    def head_type(self) -> None:
        return CustomLinearClsHead

    @pytest.fixture(autouse=True)
    def setup(self, head_type) -> None:
        self.num_classes = 2
        self.head_dim = 5
        self.default_head = head_type(self.num_classes, self.head_dim)
        self.default_input = torch.ones((2, self.head_dim))
        self.default_gt = torch.zeros(2).long()

    @e2e_pytest_unit
    def test_forward(self) -> None:
        result = self.default_head.forward_train(self.default_input, self.default_gt)
        assert "loss" in result
        assert result["loss"] >= 0

    @e2e_pytest_unit
    def test_forward_accuracy(self, head_type) -> None:
        head = head_type(self.num_classes, self.head_dim, cal_acc=True)
        result = head.forward_train(self.default_input, self.default_gt)
        assert "loss" in result
        assert "accuracy" in result
        assert "top-1" in result["accuracy"]
        assert result["accuracy"]["top-1"] >= 0

    @e2e_pytest_unit
    def test_simple_test(self) -> None:
        result = self.default_head.simple_test(self.default_input)
        assert result[0].shape[0] == self.num_classes


class TestNonLinearClsHead(TestLinearClsHead):
    @pytest.fixture(autouse=True)
    def head_type(self) -> None:
        return CustomNonLinearClsHead

    @e2e_pytest_unit
    def test_dropout(self, head_type) -> None:
        head = head_type(self.num_classes, self.head_dim, dropout=True)
        head.init_weights()
        assert len(head.classifier) > len(self.default_head.classifier)
