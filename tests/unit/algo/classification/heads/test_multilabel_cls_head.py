# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import pytest
import torch
from otx.algo.classification.heads import MultiLabelLinearClsHead, MultiLabelNonLinearClsHead
from otx.core.data.entity.base import ImageInfo
from torch import nn


@pytest.fixture()
def fxt_linear_head() -> None:
    return MultiLabelLinearClsHead(
        num_classes=3,
        in_channels=5,
    )


@pytest.fixture()
def fxt_non_linear_head() -> None:
    return MultiLabelNonLinearClsHead(
        num_classes=3,
        in_channels=5,
        hid_channels=10,
        activation=nn.PReLU(),
    )


@pytest.fixture()
def fxt_data_sample() -> dict:
    return {
        "labels": torch.Tensor([[1, 1, 1], [1, 1, 1]]),
        "imgs_info": [
            ImageInfo(
                img_idx=i,
                scale_factor=(0.448, 0.797153024911032),
                ori_shape=(281, 500),
                img_shape=(224, 224),
                ignored_labels=[],
            )
            for i in range(2)
        ],
    }


@pytest.fixture()
def fxt_data_sample_with_ignore_labels() -> dict:
    return {
        "labels": torch.Tensor([[1, 1, -1], [1, 1, -1]]),
        "imgs_info": [
            ImageInfo(
                img_idx=i,
                scale_factor=(0.448, 0.797153024911032),
                ori_shape=(281, 500),
                img_shape=(224, 224),
                ignored_labels=[2],
            )
            for i in range(2)
        ],
    }


class TestMultiLabelClsHead:
    @pytest.fixture(params=["fxt_linear_head", "fxt_non_linear_head"])
    def fxt_multilabel_head(self, request) -> nn.Module:
        return request.getfixturevalue(request.param)

    def test_predict(
        self,
        fxt_multilabel_head,
        fxt_data_sample,
    ) -> None:
        dummy_input = (torch.ones((2, 5)),)
        result = fxt_multilabel_head.predict(dummy_input, **fxt_data_sample)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)
