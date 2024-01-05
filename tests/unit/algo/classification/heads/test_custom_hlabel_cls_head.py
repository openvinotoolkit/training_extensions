# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from mmpretrain.structures import DataSample
from otx.algo.classification.heads.custom_hlabel_cls_head import CustomHierarchicalClsHead
from otx.core.data.entity.classification import HLabelInfo
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

    @pytest.fixture()
    def fxt_data_sample(self) -> DataSample:
        data_sample = DataSample(
            img_shape=(24, 24, 3),
            gt_label=torch.zeros(6, dtype=torch.long),
            hlabel_info=HLabelInfo(
                num_multiclass_heads=3,
                num_multilabel_classes=0,
                head_idx_to_logits_range={"0": (0, 2), "1": (2, 4), "2": (4, 6)},
                num_single_label_classes=6,
                empty_multiclass_head_indices=[],
                class_to_group_idx={
                    "0": (0, 0),
                    "1": (0, 1),
                    "2": (1, 0),
                    "3": (1, 1),
                    "4": (2, 0),
                    "5": (2, 1),
                },
                all_groups=[["0", "1"], ["2", "3"], ["4", "5"]],
                label_to_idx={
                    "0": 0,
                    "1": 1,
                    "2": 2,
                    "3": 3,
                    "4": 4,
                    "5": 5,
                },
            ),
        )
        return [data_sample, data_sample]

    def test_loss(self, fxt_hlabel_head, fxt_data_sample) -> None:
        dummy_input = (torch.ones((2, 24)), torch.ones((2, 24)))
        result = fxt_hlabel_head.loss(dummy_input, fxt_data_sample)
        assert "loss" in result

    def test_predict(self, fxt_hlabel_head, fxt_data_sample) -> None:
        dummy_input = (torch.ones((2, 24)), torch.ones((2, 24)))
        result = fxt_hlabel_head.predict(dummy_input, fxt_data_sample)
        assert isinstance(result[0], DataSample)
