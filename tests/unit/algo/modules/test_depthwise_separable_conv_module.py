# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Test of DepthwiseSeparableConvModule.

Reference: https://github.com/open-mmlab/mmcv/blob/main/tests/test_cnn/test_depthwise_seperable_conv_module.py
"""

import torch
from otx.algo.modules.depthwise_separable_conv_module import DepthwiseSeparableConvModule
from torch import nn


class TestDepthwiseSeparableConvModule:
    def test_forward_with_default_config(self) -> None:
        # test default config
        conv = DepthwiseSeparableConvModule(3, 8, 2)
        assert conv.depthwise_conv.conv.groups == 3
        assert conv.pointwise_conv.conv.kernel_size == (1, 1)
        assert not conv.depthwise_conv.with_norm
        assert not conv.pointwise_conv.with_norm
        assert conv.depthwise_conv.activate.__class__.__name__ == "ReLU"
        assert conv.pointwise_conv.activate.__class__.__name__ == "ReLU"
        x = torch.rand(1, 3, 256, 256)
        output = conv(x)
        assert output.shape == (1, 8, 255, 255)

    def test_forward_with_dw_norm_cfg(self) -> None:
        # test dw_norm_cfg
        conv = DepthwiseSeparableConvModule(3, 8, 2, dw_norm_cfg={"type": "BN"})
        assert conv.depthwise_conv.norm_name == "bn"
        assert not conv.pointwise_conv.with_norm
        x = torch.rand(1, 3, 256, 256)
        output = conv(x)
        assert output.shape == (1, 8, 255, 255)

    def test_forward_with_pw_norm_cfg(self) -> None:
        # test pw_norm_cfg
        conv = DepthwiseSeparableConvModule(3, 8, 2, pw_norm_cfg={"type": "BN"})
        assert not conv.depthwise_conv.with_norm
        assert conv.pointwise_conv.norm_name == "bn"
        x = torch.rand(1, 3, 256, 256)
        output = conv(x)
        assert output.shape == (1, 8, 255, 255)

    def test_forward_with_norm_cfg(self) -> None:
        # test norm_cfg
        conv = DepthwiseSeparableConvModule(3, 8, 2, norm_cfg={"type": "BN"})
        assert conv.depthwise_conv.norm_name == "bn"
        assert conv.pointwise_conv.norm_name == "bn"
        x = torch.rand(1, 3, 256, 256)
        output = conv(x)
        assert output.shape == (1, 8, 255, 255)

    def test_forward_for_order_with_norm_conv_act(self) -> None:
        # add test for ['norm', 'conv', 'act']
        conv = DepthwiseSeparableConvModule(3, 8, 2, order=("norm", "conv", "act"))
        x = torch.rand(1, 3, 256, 256)
        output = conv(x)
        assert output.shape == (1, 8, 255, 255)

        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, with_spectral_norm=True)
        assert hasattr(conv.depthwise_conv.conv, "weight_orig")
        assert hasattr(conv.pointwise_conv.conv, "weight_orig")
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)

        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, padding_mode="reflect")
        assert isinstance(conv.depthwise_conv.padding_layer, nn.ReflectionPad2d)
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)

    def test_forward_with_dw_act_cfg(self) -> None:
        # test dw_act_cfg
        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, dw_act_cfg={"type": "LeakyReLU"})
        x = torch.rand(1, 3, 256, 256)
        assert conv.depthwise_conv.activate.__class__.__name__ == "LeakyReLU"
        assert conv.pointwise_conv.activate.__class__.__name__ == "ReLU"
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)

    def test_forward_with_pw_act_cfg(self) -> None:
        # test pw_act_cfg
        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, pw_act_cfg={"type": "LeakyReLU"})
        x = torch.rand(1, 3, 256, 256)
        assert conv.depthwise_conv.activate.__class__.__name__ == "ReLU"
        assert conv.pointwise_conv.activate.__class__.__name__ == "LeakyReLU"
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)

    def test_forward_with_act_cfg(self) -> None:
        # test act_cfg
        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, act_cfg={"type": "LeakyReLU"})
        x = torch.rand(1, 3, 256, 256)
        assert conv.depthwise_conv.activate.__class__.__name__ == "LeakyReLU"
        assert conv.pointwise_conv.activate.__class__.__name__ == "LeakyReLU"
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)
