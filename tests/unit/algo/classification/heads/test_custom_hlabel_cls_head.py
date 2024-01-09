# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from mmpretrain.structures import DataSample
from otx.algo.classification.heads.custom_hlabel_cls_head import CustomHierarchicalClsHead
from torch import nn


class TestCustomHierarchicalClsHead:
    @pytest.fixture()
    def fxt_hlabel_head(self) -> nn.Module:
        return CustomHierarchicalClsHead(
            num_multiclass_heads=3,
            num_multilabel_classes=0,
            in_channels=24,
            num_classes=6,
            multiclass_loss_cfg={
                "type": "CrossEntropyLoss",
                "use_sigmoid": False,
                "reduction": "mean",
                "loss_weight": 1.0,
            },
        )

    def test_loss(self, fxt_hlabel_head, fxt_data_sample, fxt_hlabel_info) -> None:
        fxt_hlabel_head.set_hlabel_info(fxt_hlabel_info)

        dummy_input = (torch.ones((2, 24)), torch.ones((2, 24)))
        result = fxt_hlabel_head.loss(dummy_input, fxt_data_sample)
        assert "loss" in result

    def test_predict(self, fxt_hlabel_head, fxt_data_sample, fxt_hlabel_info) -> None:
        fxt_hlabel_head.set_hlabel_info(fxt_hlabel_info)

        dummy_input = (torch.ones((2, 24)), torch.ones((2, 24)))
        result = fxt_hlabel_head.predict(dummy_input, fxt_data_sample)
        assert isinstance(result[0], DataSample)
