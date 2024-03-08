# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of MoViNet Head."""


import pytest
import torch
from otx.algo.action_classification.heads.movinet_head import MoViNetHead


class TestMoViNetHead:
    @pytest.fixture()
    def fxt_movinet_head(self) -> MoViNetHead:
        return MoViNetHead(
            5,
            480,
            2048,
            {"type": "CrossEntropyLoss", "loss_weight": 1.0},
            average_clips="prob",
        )

    def test_forward(self, fxt_movinet_head: MoViNetHead) -> None:
        fxt_movinet_head.init_weights()
        x = torch.randn(5, 480, 1, 1, 1)
        assert fxt_movinet_head(x).shape == torch.Size([5, 5])
