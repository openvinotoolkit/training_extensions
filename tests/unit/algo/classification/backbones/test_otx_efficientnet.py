# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.algo.classification.backbones.otx_efficientnet import OTXEfficientNet, calc_tf_padding


class TestOTXEfficientNet:
    @pytest.mark.parametrize("version", ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"])
    def test_forward(self, version) -> None:
        model = OTXEfficientNet(version, pretrained=None)
        assert model(torch.randn(1, 3, 244, 244))[0].shape[-1] == 8
        assert model(torch.randn(1, 3, 244, 244))[0].shape[-2] == 8

        assert isinstance(model(torch.randn(1, 3, 244, 244), return_featuremaps=True), tuple)
        assert model(torch.randn(1, 3, 244, 244), get_embeddings=True)[0].shape[-1] == 8

    def test_calc_tf_padding(self) -> None:
        x = torch.randn((1, 3, 32, 32))
        kernel_size = 3
        assert calc_tf_padding(x, kernel_size) == (1, 1, 1, 1)
