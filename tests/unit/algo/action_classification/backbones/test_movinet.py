# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of MoViNet backbone."""

import pytest
import torch
from otx.algo.action_classification.backbones.movinet import MoViNetBackbone


class TestMoViNet:
    @pytest.fixture()
    def fxt_movinet(self) -> MoViNetBackbone:
        return MoViNetBackbone()

    def test_forward(self, fxt_movinet: MoViNetBackbone) -> None:
        x = torch.randn(1, 3, 8, 224, 224)
        assert fxt_movinet(x).shape == torch.Size([1, 480, 1, 1, 1])
