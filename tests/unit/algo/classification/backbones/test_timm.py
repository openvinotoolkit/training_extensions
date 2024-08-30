# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path

import torch
from otx.algo.classification.backbones.timm import TimmBackbone


class TestOTXEfficientNetV2:
    def test_forward(self):
        model = TimmBackbone(backbone="tf_efficientnetv2_s.in21k")
        assert model(torch.randn(1, 3, 244, 244))[0].shape == torch.Size([1, 1280, 8, 8])

    def test_get_config_optim(self):
        model = TimmBackbone(backbone="tf_efficientnetv2_s.in21k")
        assert model.get_config_optim([0.01])[0]["lr"] == 0.01
        assert model.get_config_optim(0.01)[0]["lr"] == 0.01

    def test_check_pretrained_weight_download(self):
        target = Path(os.environ.get("HF_HUB_CACHE")) / "models--timm--tf_efficientnetv2_s.in21k"
        if target.exists():
            shutil.rmtree(target)
        assert not target.exists()
        TimmBackbone(backbone="tf_efficientnetv2_s.in21k", pretrained=True)
        assert target.exists()
