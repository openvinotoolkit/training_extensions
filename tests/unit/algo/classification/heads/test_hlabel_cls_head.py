# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import Any

import pytest
import torch
from otx.algo.classification.heads import (
    HierarchicalCBAMClsHead,
    HierarchicalLinearClsHead,
    HierarchicalNonLinearClsHead,
)
from otx.algo.classification.heads.hlabel_cls_head import CBAM, ChannelAttention, SpatialAttention
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


class TestHierarchicalLinearClsHead:
    @pytest.fixture()
    def fxt_head_attrs(self, fxt_hlabel_cifar) -> dict[str, Any]:
        return {
            **fxt_hlabel_cifar.as_head_config_dict(),
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

    @pytest.fixture()
    def fxt_hlabel_cbam_head(self, fxt_head_attrs) -> nn.Module:
        fxt_head_attrs["step_size"] = 1
        return HierarchicalCBAMClsHead(**fxt_head_attrs)

    @pytest.fixture(params=["fxt_hlabel_linear_head", "fxt_hlabel_non_linear_head", "fxt_hlabel_cbam_head"])
    def fxt_hlabel_head(self, request) -> nn.Module:
        return request.getfixturevalue(request.param)

    def test_predict(
        self,
        fxt_hlabel_head,
        fxt_data_sample,
    ) -> None:
        dummy_input = (torch.ones((2, 24)), torch.ones((2, 24)))
        result = fxt_hlabel_head.predict(dummy_input, **fxt_data_sample)
        assert isinstance(result, dict)
        assert "scores" in result
        assert result["scores"].shape == (2, 3)
        assert "labels" in result
        assert result["labels"].shape == (2, 3)


class TestChannelAttention:
    @pytest.fixture()
    def fxt_channel_attention(self) -> ChannelAttention:
        return ChannelAttention(in_channels=64, reduction=16)

    def test_forward(self, fxt_channel_attention) -> None:
        input_tensor = torch.rand((8, 64, 32, 32))
        result = fxt_channel_attention(input_tensor)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)


class TestSpatialAttention:
    @pytest.fixture()
    def fxt_spatial_attention(self) -> SpatialAttention:
        return SpatialAttention(kernel_size=7)

    def test_forward(self, fxt_spatial_attention) -> None:
        input_tensor = torch.rand((8, 64, 32, 32))
        result = fxt_spatial_attention(input_tensor)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)


class TestCBAM:
    @pytest.fixture()
    def fxt_cbam(self) -> CBAM:
        return CBAM(in_channels=64, reduction=16, kernel_size=7)

    def test_forward(self, fxt_cbam) -> None:
        input_tensor = torch.rand((8, 64, 32, 32))
        result = fxt_cbam(input_tensor)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)


class TestHierarchicalCBAMClsHead:
    @pytest.fixture()
    def fxt_hierarchical_cbam_cls_head(self) -> HierarchicalCBAMClsHead:
        head_idx_to_logits_range = {"0": (0, 5), "1": (5, 10), "2": (10, 12)}
        return HierarchicalCBAMClsHead(
            num_multiclass_heads=3,
            num_multilabel_classes=0,
            head_idx_to_logits_range=head_idx_to_logits_range,
            num_single_label_classes=12,
            empty_multiclass_head_indices=[],
            in_channels=64,
            num_classes=12,
            multiclass_loss=CrossEntropyLoss(),
            multilabel_loss=None,
        )

    def test_forward(self, fxt_hierarchical_cbam_cls_head) -> None:
        input_tensor = torch.rand((8, 64, 7, 7))
        result = fxt_hierarchical_cbam_cls_head(input_tensor)
        assert result.shape == (8, 12)

    def test_pre_logits(self, fxt_hierarchical_cbam_cls_head) -> None:
        input_tensor = torch.rand((8, 64, 7, 7))
        pre_logits = fxt_hierarchical_cbam_cls_head.pre_logits(input_tensor)
        assert pre_logits.shape == (8, 64 * 7 * 7)

    def test_pre_logits_tuple_step_size(self) -> None:
        head_idx_to_logits_range = {"0": (0, 5), "1": (5, 10), "2": (10, 12)}
        head = HierarchicalCBAMClsHead(
            num_multiclass_heads=3,
            num_multilabel_classes=0,
            head_idx_to_logits_range=head_idx_to_logits_range,
            num_single_label_classes=12,
            empty_multiclass_head_indices=[],
            in_channels=64,
            num_classes=12,
            multiclass_loss=CrossEntropyLoss(),
            multilabel_loss=None,
            step_size=(14, 7),
        )

        input_tensor = torch.rand((8, 64, 14, 7))
        pre_logits = head.pre_logits(input_tensor)
        assert pre_logits.shape == (8, 64 * 14 * 7)
