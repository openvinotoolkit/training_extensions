# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of ChannelAttention."""

import pytest
import torch
from otx.algo.detection.layers import ChannelAttention
from torch.nn import AdaptiveAvgPool2d, Conv2d, Hardsigmoid


class TestChannelAttention:
    @pytest.fixture()
    def channel_attention(self) -> ChannelAttention:
        return ChannelAttention(3)

    def test_init(self, channel_attention) -> None:
        """Test __init__."""
        assert isinstance(channel_attention.global_avgpool, AdaptiveAvgPool2d)
        assert isinstance(channel_attention.fc, Conv2d)
        assert isinstance(channel_attention.act, Hardsigmoid)

    def test_forward(self, channel_attention) -> None:
        """Test forward."""
        x = torch.randn(1, 3, 10, 10)
        outs = channel_attention(x)

        assert outs.shape == torch.Size([1, 3, 10, 10])
