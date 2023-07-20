# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.heads.custom_hierarchical_linear_cls_head import (
    CustomHierarchicalLinearClsHead,
)
from otx.algorithms.classification.adapters.mmcls.models.heads.custom_hierarchical_non_linear_cls_head import (
    CustomHierarchicalNonLinearClsHead,
)
from otx.algorithms.classification.adapters.mmcls.models.losses.asymmetric_loss_with_ignore import (
    AsymmetricLossWithIgnore,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomHierarchicalLinearClsHead:
    @pytest.fixture(autouse=True)
    def head_type(self) -> None:
        return CustomHierarchicalLinearClsHead

    @pytest.fixture(autouse=True)
    def setup(self, head_type) -> None:
        self.num_classes = 6
        self.head_dim = 10
        self.cls_heads_info = {
            "num_multiclass_heads": 3,
            "num_multilabel_classes": 0,
            "head_idx_to_logits_range": {"0": (0, 2), "1": (2, 4), "2": (4, 6)},
            "num_single_label_classes": 6,
            "empty_multiclass_head_indices": [],
        }
        self.loss = dict(type="CrossEntropyLoss", use_sigmoid=False, reduction="mean", loss_weight=1.0)
        self.multilabel_loss = dict(type=AsymmetricLossWithIgnore.__name__, reduction="sum")
        self.default_head = head_type(
            self.num_classes,
            self.head_dim,
            hierarchical_info=self.cls_heads_info,
            loss=self.loss,
            multilabel_loss=self.multilabel_loss,
        )
        self.default_head.init_weights()
        self.default_input = torch.ones((2, self.head_dim))
        self.default_gt = torch.zeros((2, 3))

    @e2e_pytest_unit
    def test_forward(self) -> None:
        result = self.default_head.forward_train(self.default_input, self.default_gt)
        assert "loss" in result
        assert result["loss"] >= 0 and not torch.isnan(result["loss"])

        empty_head_gt_full = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        result_include_empty_full = self.default_head.forward_train(self.default_input, empty_head_gt_full)
        assert "loss" in result_include_empty_full
        assert result_include_empty_full["loss"] >= 0 and not torch.isnan(result_include_empty_full["loss"])

        empty_head_gt_partial = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        result_include_empty_partial = self.default_head.forward_train(self.default_input, empty_head_gt_partial)
        assert "loss" in result_include_empty_partial
        assert result_include_empty_partial["loss"] >= 0 and not torch.isnan(result_include_empty_partial["loss"])

    @e2e_pytest_unit
    def test_simple_test(self) -> None:
        result = self.default_head.simple_test(self.default_input)
        assert result[0].shape[0] == self.num_classes

    @e2e_pytest_unit
    def test_zero_classes(self, head_type) -> None:
        self.cls_heads_info["num_multiclass_heads"] = 0
        self.cls_heads_info["num_multilabel_classes"] = 0
        with pytest.raises(ValueError):
            head_type(
                self.num_classes,
                self.head_dim,
                hierarchical_info=self.cls_heads_info,
                loss=self.loss,
                multilabel_loss=self.multilabel_loss,
            )

    @e2e_pytest_unit
    def test_neg_classes(self, head_type) -> None:
        with pytest.raises(ValueError):
            head_type(
                -1,
                self.head_dim,
                hierarchical_info=self.cls_heads_info,
                loss=self.loss,
                multilabel_loss=self.multilabel_loss,
            )


class TestCustomHierarchicalNonLinearClsHead(TestCustomHierarchicalLinearClsHead):
    @pytest.fixture(autouse=True)
    def head_type(self) -> None:
        return CustomHierarchicalNonLinearClsHead

    @e2e_pytest_unit
    def test_dropout(self, head_type) -> None:
        head = head_type(self.num_classes, self.head_dim, dropout=True, hierarchical_info=self.cls_heads_info)
        head.init_weights()
        assert len(head.classifier) > len(self.default_head.classifier)
