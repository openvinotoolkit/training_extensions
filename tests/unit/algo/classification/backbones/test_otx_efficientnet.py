# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from otx.algo.classification.backbones.efficientnet import OTXEfficientNet


class TestOTXEfficientNet:
    @pytest.mark.parametrize("version", ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"])
    def test_forward(self, version):
        model = OTXEfficientNet(version, pretrained=None)
        assert model(torch.randn(1, 3, 244, 244))[0].shape[-1] == 8
        assert model(torch.randn(1, 3, 244, 244))[0].shape[-2] == 8

    def test_set_input_size(self):
        input_size = (300, 300)
        model = OTXEfficientNet("b0", input_size=input_size, pretrained=None)
        assert model.in_size == input_size
