from __future__ import annotations

from functools import partial
from typing import Any

import pytest
import torch
from otx.algo.modules.norm import build_norm_layer
from otx.algo.segmentation.heads.ham_head import NNLightHamHead
from torch import nn


class TestNNLightHamHead:
    @pytest.fixture()
    def head_config(self) -> dict[str, Any]:
        return {
            "ham_kwargs": {"md_r": 16, "md_s": 1, "eval_steps": 7, "train_steps": 6},
            "in_channels": [128, 320, 512],
            "in_index": [1, 2, 3],
            "normalization": partial(build_norm_layer, nn.GroupNorm, num_groups=32, requires_grad=True),
            "align_corners": False,
            "channels": 512,
            "dropout_ratio": 0.1,
            "ham_channels": 512,
            "num_classes": 2,
        }

    def test_init(self, head_config):
        light_ham_head = NNLightHamHead(**head_config)
        assert light_ham_head.ham_channels == head_config["ham_channels"]

    @pytest.fixture()
    def batch_size(self) -> int:
        return 8

    @pytest.fixture()
    def fake_input(self, batch_size) -> list[torch.Tensor]:
        return [
            torch.rand(batch_size, 64, 128, 128),
            torch.rand(batch_size, 128, 64, 64),
            torch.rand(batch_size, 320, 32, 32),
            torch.rand(batch_size, 512, 16, 16),
        ]

    def test_forward(self, head_config, fake_input, batch_size):
        light_ham_head = NNLightHamHead(**head_config)
        out = light_ham_head.forward(fake_input)
        assert out.size()[0] == batch_size
        assert out.size()[2] == fake_input[head_config["in_index"][0]].size()[2]
        assert out.size()[3] == fake_input[head_config["in_index"][0]].size()[3]
