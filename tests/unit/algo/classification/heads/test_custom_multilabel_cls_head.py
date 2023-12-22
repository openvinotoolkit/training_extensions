# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import pytest
import torch
from mmpretrain.structures import DataSample
from otx.algo.classification.heads import CustomMultiLabelLinearClsHead, CustomMultiLabelNonLinearClsHead
from otx.algo.classification.losses import AsymmetricAngularLossWithIgnore


@pytest.fixture()
def fxt_linear_head() -> None:
    return CustomMultiLabelLinearClsHead(
        num_classes=2,
        in_channels=5,
        loss={
            "type": AsymmetricAngularLossWithIgnore.__name__,
            "reduction": "sum",
        },
    )


@pytest.fixture()
def fxt_non_linear_head() -> None:
    return CustomMultiLabelNonLinearClsHead(
        num_classes=2,
        in_channels=5,
        hid_channels=10,
        act_cfg={"type": "PReLU"},
        loss={
            "type": AsymmetricAngularLossWithIgnore.__name__,
            "reduction": "sum",
        },
    )


@pytest.fixture()
def fxt_data_sample() -> None:
    return DataSample(
        img_shape=(224, 224, 3),
        gt_label=torch.tensor([0, 1]),
    )


class TestCustomMultiLabelClsHead:
    def test_linear_loss(self, fxt_linear_head, fxt_data_sample) -> None:
        inputs = (torch.ones((2, 5)),)

        result = fxt_linear_head.loss(inputs, [fxt_data_sample, fxt_data_sample])
        assert "loss" in result
        assert result["loss"] >= 0

    def test_nonlinear_loss(self, fxt_non_linear_head, fxt_data_sample) -> None:
        inputs = (torch.ones((2, 5)),)

        result = fxt_non_linear_head.loss(inputs, [fxt_data_sample, fxt_data_sample])
        assert "loss" in result
        assert result["loss"] >= 0
