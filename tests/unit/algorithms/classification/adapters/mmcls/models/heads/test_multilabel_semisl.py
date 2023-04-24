# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.heads.semisl_multilabel_cls_head import (
    SemiLinearMultilabelClsHead,
    SemiNonLinearMultilabelClsHead,
)
from otx.algorithms.classification.adapters.mmcls.models.losses.asymmetric_loss_with_ignore import (
    AsymmetricLossWithIgnore,
)
from otx.algorithms.classification.adapters.mmcls.models.losses.barlowtwins_loss import (
    BarlowTwinsLoss,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSemiLinearMultilabelClsHead:
    @pytest.fixture(autouse=True)
    def head_type(self):
        return SemiLinearMultilabelClsHead

    @pytest.fixture(autouse=True)
    def setup(self, head_type) -> None:
        self.num_classes = 2
        self.head_dim = 5
        self.loss = dict(type=AsymmetricLossWithIgnore.__name__, reduction="sum")
        self.aux_loss = dict(type=BarlowTwinsLoss.__name__, off_diag_penality=1.0 / 128.0, loss_weight=1.0)
        self.default_head = head_type(self.num_classes, self.head_dim, loss=self.loss)
        self.default_head.init_weights()
        self.default_input = torch.ones((2, self.head_dim))
        self.default_ssl_input = {
            "labeled_weak": self.default_input,
            "labeled_strong": self.default_input,
            "unlabeled_weak": self.default_input,
            "unlabeled_strong": self.default_input,
        }
        self.default_gt = torch.zeros((2, self.num_classes))

    @e2e_pytest_unit
    def test_forward(self) -> None:
        result = self.default_head.forward_train(self.default_ssl_input, self.default_gt)
        assert "loss" in result
        assert "unlabeled_loss" in result
        assert result["loss"] >= 0

    @e2e_pytest_unit
    def test_simple_test(self) -> None:
        result = self.default_head.simple_test(self.default_input)
        assert result[0].shape[0] == self.num_classes

    @e2e_pytest_unit
    def test_angular(self, head_type) -> None:
        head = head_type(self.num_classes, self.head_dim, loss=self.loss, normalized=True)
        result = head.simple_test(self.default_input)
        assert result[0].shape[0] == self.num_classes

    @e2e_pytest_unit
    def test_neg_classes(self, head_type) -> None:
        with pytest.raises(ValueError):
            head_type(-1, self.head_dim, loss=self.loss, normalized=True)


class TestSemiNonLinearMultilabelClsHead(TestSemiLinearMultilabelClsHead):
    @pytest.fixture(autouse=True)
    def head_type(self):
        return SemiNonLinearMultilabelClsHead

    @e2e_pytest_unit
    def test_dropout(self, head_type) -> None:
        head = head_type(self.num_classes, self.head_dim, dropout=True)
        head.init_weights()
        assert len(head.classifier) > len(self.default_head.classifier)
