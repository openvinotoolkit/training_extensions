# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import Any

import pytest
import torch
from mmpretrain.structures import DataSample
from otx.algo.classification.heads.custom_hlabel_linear_cls_head import CustomHierarchicalLinearClsHead
from otx.algo.classification.heads.custom_hlabel_non_linear_cls_head import CustomHierarchicalNonLinearClsHead
from torch import nn


@pytest.fixture()
def fxt_data_sample() -> DataSample:
    data_sample = DataSample(
        gt_label=torch.ones(6, dtype=torch.long),
        metainfo={
            "img_shape": (24, 24, 3),
            "ignored_labels": None,
        },
    )
    return [data_sample] * 18


@pytest.fixture()
def fxt_data_sample_with_ignored_labels() -> DataSample:
    data_sample = DataSample(
        gt_label=torch.ones(6, dtype=torch.long),
        metainfo={
            "img_shape": (24, 24, 3),
            "ignored_labels": [5],
        },
    )
    return [data_sample] * 18


class TestCustomHierarchicalLinearClsHead:
    @pytest.fixture()
    def fxt_head_attrs(self, fxt_hlabel_multilabel_info) -> dict[str, Any]:
        return {
            **fxt_hlabel_multilabel_info.as_head_config_dict(),
            "in_channels": 24,
            "multiclass_loss_cfg": {
                "type": "CrossEntropyLoss",
                "use_sigmoid": False,
                "reduction": "mean",
                "loss_weight": 1.0,
            },
            "multilabel_loss_cfg": {
                "reduction": "sum",
                "gamma_neg": 1.0,
                "gamma_pos": 0.0,
                "type": "AsymmetricAngularLossWithIgnore",
            },
        }

    @pytest.fixture()
    def fxt_hlabel_linear_head(self, fxt_head_attrs) -> nn.Module:
        return CustomHierarchicalLinearClsHead(**fxt_head_attrs)

    @pytest.fixture()
    def fxt_hlabel_non_linear_head(self, fxt_head_attrs) -> nn.Module:
        return CustomHierarchicalNonLinearClsHead(**fxt_head_attrs)

    @pytest.fixture(params=["fxt_hlabel_linear_head", "fxt_hlabel_non_linear_head"])
    def fxt_hlabel_head(self, request) -> nn.Module:
        return request.getfixturevalue(request.param)

    def test_loss(
        self,
        fxt_hlabel_head,
        fxt_data_sample,
        fxt_data_sample_with_ignored_labels,
    ) -> None:
        dummy_input = (torch.ones((18, 24)), torch.ones((18, 24)))
        result_without_ignored_labels = fxt_hlabel_head.loss(dummy_input, fxt_data_sample)

        result_with_ignored_labels = fxt_hlabel_head.loss(
            dummy_input,
            fxt_data_sample_with_ignored_labels,
        )
        assert result_with_ignored_labels["loss"] <= result_without_ignored_labels["loss"]

    def test_predict(
        self,
        fxt_hlabel_head,
        fxt_data_sample,
    ) -> None:
        dummy_input = (torch.ones((2, 24)), torch.ones((2, 24)))
        result = fxt_hlabel_head.predict(dummy_input, fxt_data_sample)
        assert isinstance(result[0], DataSample)
