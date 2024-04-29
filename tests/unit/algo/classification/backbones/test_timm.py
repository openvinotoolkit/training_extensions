# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from otx.algo.classification.backbones.timm import TimmBackbone


class TestOTXEfficientNetV2:
    def test_forward(self):
        model = TimmBackbone(backbone="efficientnetv2_s_21k")
        model.init_weights()
        assert model(torch.randn(1, 3, 244, 244))[0].shape == torch.Size([1, 1280, 8, 8])

    def test_get_config_optim(self):
        model = TimmBackbone(backbone="efficientnetv2_s_21k")
        assert model.get_config_optim([0.01])[0]["lr"] == 0.01
        assert model.get_config_optim(0.01)[0]["lr"] == 0.01
