# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from otx.algo.classification.backbones.otx_efficientnet_v2 import OTXEfficientNetV2


class TestOTXEfficientNetV2:
    def test_forward(self):
        model = OTXEfficientNetV2()
        model.init_weights()
        assert model(torch.randn(1, 3, 244, 244))[0].shape == torch.Size([1, 1280, 8, 8])

    def test_get_config_optim(self):
        model = OTXEfficientNetV2()
        assert model.get_config_optim([0.01])[0]["lr"] == 0.01
        assert model.get_config_optim(0.01)[0]["lr"] == 0.01
