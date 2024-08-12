# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of CSPLayer."""

import torch
from otx.algo.detection.layers import ChannelAttention
from otx.algo.detection.layers.csp_layer import CSPLayer, CSPNeXtBlock, DarknetBottleneck
from otx.algo.modules.activation import Swish
from otx.algo.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from torch.nn import BatchNorm2d, Conv2d


class TestCSPLayer:
    def test_init(self) -> None:
        """Test __init__."""
        csp_layer = CSPLayer(3, 5)

        assert isinstance(csp_layer.blocks[0], DarknetBottleneck)
        assert isinstance(csp_layer.blocks[0].conv2, Conv2dModule)
        assert isinstance(csp_layer.blocks[0].conv1.conv, Conv2d)
        assert isinstance(csp_layer.blocks[0].conv1.bn, BatchNorm2d)
        assert isinstance(csp_layer.blocks[0].conv1.activation, Swish)
        assert not hasattr(csp_layer, "attention")

        # use DepthwiseSeparableConvModule
        csp_layer = CSPLayer(3, 5, use_depthwise=True)
        assert isinstance(csp_layer.blocks[0].conv2, DepthwiseSeparableConvModule)

        assert csp_layer.blocks[0]

        # use CSPNeXtBlock
        csp_layer = CSPLayer(3, 5, use_cspnext_block=True)

        assert isinstance(csp_layer.blocks[0], CSPNeXtBlock)

        # use channel_attention
        csp_layer = CSPLayer(3, 5, channel_attention=True)

        assert hasattr(csp_layer, "attention")
        assert isinstance(csp_layer.attention, ChannelAttention)

    def test_forward(self) -> None:
        """Test forward."""
        csp_layer = CSPLayer(3, 5)

        # forward
        x = torch.randn(1, 3, 10, 10)
        outs = csp_layer(x)

        assert outs.shape == torch.Size([1, 5, 10, 10])
