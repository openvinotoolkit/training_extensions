# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import pytest
import torch
from otx.algo.classification.heads import (
    CustomMultiLabelLinearClsHead,
    CustomMultiLabelNonLinearClsHead
)
from otx.algo.classification.losses import AsymmetricAngularLossWithIgnore

from mmpretrain.structures import DataSample

@pytest.fixture()
def fxt_linear_head() -> None:
    head = CustomMultiLabelLinearClsHead(
        num_classes=2,
        in_channels=5,
        loss = dict(
            type=AsymmetricAngularLossWithIgnore.__name__,
            reduction="sum"
        )
    )
    return head

@pytest.fixture()
def fxt_non_linear_head() -> None:
    head = CustomMultiLabelNonLinearClsHead(
        num_classes=2,
        in_channels=5,
        hid_channels=10,
        act_cfg=dict(type="PReLU"),
        loss = dict(
            type=AsymmetricAngularLossWithIgnore.__name__,
            reduction="sum"
        )
    )
    return head

@pytest.fixture()
def fxt_data_sample() -> None:
    data_sample = DataSample(
        img_shape=(224, 224, 3),
        gt_label=torch.tensor([0, 1])
    )
    return data_sample

class TestCustomMultiLabelClsHead:
    def test_linear_loss(self, fxt_linear_head, fxt_data_sample) -> None:
        input = (torch.ones((2, 5)), )
        
        result = fxt_linear_head.loss(input, [fxt_data_sample, fxt_data_sample])
        assert "loss" in result
        assert result["loss"] >= 0
    
    def test_nonlinear_loss(self, fxt_non_linear_head, fxt_data_sample) -> None:
        input = (torch.ones((2, 5)), )
        
        result = fxt_non_linear_head.loss(input, [fxt_data_sample, fxt_data_sample])
        assert "loss" in result
        assert result["loss"] >= 0
    