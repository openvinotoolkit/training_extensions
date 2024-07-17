# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of Presnet."""

import torch
from otx.algo.detection.backbones.presnet import PResNet, FrozenBatchNorm2d


class TestPresnet():
    def test_presnet_forward(self):
        model = PResNet(depth=50)
        input = torch.randn(1, 3, 224, 224)
        output = model(input)
        assert len(output) == 4
        assert output[0].shape == torch.Size([1, 256, 56, 56])
        assert output[1].shape == torch.Size([1, 512, 28, 28])
        assert output[2].shape == torch.Size([1, 1024, 14, 14])
        assert output[3].shape == torch.Size([1, 2048, 7, 7])

    def test_presnet_freeze_parameters(self):
        model = PResNet(depth=50, freeze_at=2)
        for name, param in model.named_parameters():
            if name.startswith('conv1') or name.startswith('res_layers.0'):
                assert not param.requires_grad

    def test_presnet_freeze_norm(self):
        model = PResNet(depth=50, freeze_norm=True)
        for name, param in model.named_parameters():
            if "norm" in name:
                assert isinstance(param, FrozenBatchNorm2d)
