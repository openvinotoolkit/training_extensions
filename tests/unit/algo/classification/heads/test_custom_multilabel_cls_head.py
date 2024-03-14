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
        num_classes=3,
        in_channels=5,
        loss={
            "type": AsymmetricAngularLossWithIgnore.__name__,
            "reduction": "sum",
        },
    )


@pytest.fixture()
def fxt_non_linear_head() -> None:
    return CustomMultiLabelNonLinearClsHead(
        num_classes=3,
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
        gt_score=torch.Tensor([1, 1, 1]),
        metainfo={
            "scale_factor": (0.448, 0.797153024911032),
            "img_shape": (224, 224),
            "pad_shape": (224, 224),
            "ignored_labels": None,
            "ori_shape": (281, 500),
        },
    )


@pytest.fixture()
def fxt_data_sample_with_ignore_labels() -> None:
    return DataSample(
        gt_score=torch.Tensor([1, 1, -1]),
        metainfo={
            "scale_factor": (0.448, 0.797153024911032),
            "img_shape": (224, 224),
            "pad_shape": (224, 224),
            "ignored_labels": [2],
            "ori_shape": (281, 500),
        },
    )


class TestCustomMultiLabelClsHead:
    def test_linear_loss(self, fxt_linear_head, fxt_data_sample, fxt_data_sample_with_ignore_labels) -> None:
        inputs = (torch.ones((2, 5)),)

        result_with_ignore_label = fxt_linear_head.loss(
            inputs,
            [fxt_data_sample_with_ignore_labels, fxt_data_sample_with_ignore_labels],
        )
        result_without_ignore_label = fxt_linear_head.loss(inputs, [fxt_data_sample, fxt_data_sample])
        assert result_with_ignore_label["loss"] < result_without_ignore_label["loss"]

    def test_nonlinear_loss(self, fxt_non_linear_head, fxt_data_sample, fxt_data_sample_with_ignore_labels) -> None:
        inputs = (torch.ones((2, 5)),)

        result_with_ignore_label = fxt_non_linear_head.loss(
            inputs,
            [fxt_data_sample_with_ignore_labels, fxt_data_sample_with_ignore_labels],
        )
        result_without_ignore_label = fxt_non_linear_head.loss(inputs, [fxt_data_sample, fxt_data_sample])
        assert result_with_ignore_label["loss"] < result_without_ignore_label["loss"]
