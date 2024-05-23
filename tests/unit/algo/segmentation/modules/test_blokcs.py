from __future__ import annotations

from typing import Any

import pytest
import torch
from otx.algo.segmentation.modules.blocks import AsymmetricPositionAttentionModule, LocalAttentionModule


class TestAsymmetricPositionAttentionModule:
    @pytest.fixture()
    def init_cfg(self) -> dict[str, Any]:
        return {
            "in_channels": 320,
            "key_channels": 128,
            "value_channels": 320,
            "psp_size": [1, 3, 6, 8],
            "conv_cfg": {"type": "Conv2d"},
            "norm_cfg": {"type": "BN"},
        }

    def test_init(self, init_cfg):
        module = AsymmetricPositionAttentionModule(**init_cfg)

        assert module.in_channels == init_cfg["in_channels"]
        assert module.key_channels == init_cfg["key_channels"]
        assert module.value_channels == init_cfg["value_channels"]
        assert module.conv_cfg == init_cfg["conv_cfg"]
        assert module.norm_cfg == init_cfg["norm_cfg"]

    @pytest.fixture()
    def fake_input(self) -> torch.Tensor:
        return torch.rand(8, 320, 16, 16)

    def test_forward(self, init_cfg, fake_input):
        module = AsymmetricPositionAttentionModule(**init_cfg)
        out = module.forward(fake_input)

        assert out.size() == fake_input.size()


class TestLocalAttentionModule:
    @pytest.fixture()
    def init_cfg(self) -> dict[str, Any]:
        return {
            "num_channels": 320,
            "conv_cfg": {"type": "Conv2d"},
            "norm_cfg": {"type": "BN"},
        }

    def test_init(self, init_cfg):
        module = LocalAttentionModule(**init_cfg)

        assert module.num_channels == init_cfg["num_channels"]
        assert module.conv_cfg == init_cfg["conv_cfg"]
        assert module.norm_cfg == init_cfg["norm_cfg"]

    @pytest.fixture()
    def fake_input(self) -> torch.Tensor:
        return torch.rand(8, 320, 16, 16)

    def test_forward(self, init_cfg, fake_input):
        module = LocalAttentionModule(**init_cfg)

        out = module.forward(fake_input)
        assert out.size() == fake_input.size()
