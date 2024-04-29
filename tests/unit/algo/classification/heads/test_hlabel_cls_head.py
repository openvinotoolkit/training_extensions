# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import Any

import pytest
import torch
from otx.algo.classification.heads import HierarchicalLinearClsHead, HierarchicalNonLinearClsHead
from otx.algo.classification.losses import AsymmetricAngularLossWithIgnore
from otx.core.data.entity.base import ImageInfo
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss


@pytest.fixture()
def fxt_data_sample() -> dict:
    return {
        "labels": torch.ones((18, 6), dtype=torch.long),
        "imgs_info": [
            ImageInfo(
                img_idx=i,
                ori_shape=(24, 24, 3),
                img_shape=(24, 24, 3),
                ignored_labels=[],
            )
            for i in range(18)
        ],
    }


@pytest.fixture()
def fxt_data_sample_with_ignored_labels() -> dict:
    return {
        "labels": torch.ones((18, 6), dtype=torch.long),
        "imgs_info": [
            ImageInfo(
                img_idx=i,
                ori_shape=(24, 24, 3),
                img_shape=(24, 24, 3),
                ignored_labels=[3],
            )
            for i in range(18)
        ],
    }


class TestHierarchicalLinearClsHead:
    @pytest.fixture()
    def fxt_head_attrs(self, fxt_hlabel_multilabel_info) -> dict[str, Any]:
        return {
            **fxt_hlabel_multilabel_info.as_head_config_dict(),
            "in_channels": 24,
            "multiclass_loss": CrossEntropyLoss(),
            "multilabel_loss": AsymmetricAngularLossWithIgnore(),
        }

    @pytest.fixture()
    def fxt_hlabel_linear_head(self, fxt_head_attrs) -> nn.Module:
        return HierarchicalLinearClsHead(**fxt_head_attrs)

    @pytest.fixture()
    def fxt_hlabel_non_linear_head(self, fxt_head_attrs) -> nn.Module:
        return HierarchicalNonLinearClsHead(**fxt_head_attrs)

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
        result_without_ignored_labels = fxt_hlabel_head.loss(dummy_input, **fxt_data_sample)

        result_with_ignored_labels = fxt_hlabel_head.loss(
            dummy_input,
            **fxt_data_sample_with_ignored_labels,
        )
        assert result_with_ignored_labels <= result_without_ignored_labels

    def test_predict(
        self,
        fxt_hlabel_head,
        fxt_data_sample,
    ) -> None:
        dummy_input = (torch.ones((2, 24)), torch.ones((2, 24)))
        result = fxt_hlabel_head.predict(dummy_input, **fxt_data_sample)
        assert isinstance(result, dict)
        assert "scores" in result
        assert result["scores"].shape == (2, 6)
        assert "labels" in result
        assert result["labels"].shape == (2, 6)
