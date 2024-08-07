# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmcv/blob/main/tests/test_cnn/test_conv_module.py

from functools import partial

import pytest
import torch
from otx.algo.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from torch import nn


def test_conv_module():
    norm_cfg = "norm"
    with pytest.raises(AssertionError):
        # norm_cfg must be a dict or None
        Conv2dModule(3, 8, 2, norm_cfg=norm_cfg)

    activation_callable = nn.Softmax
    with pytest.raises(ValueError, match="Unsupported activation"):
        # softmax is not supported
        Conv2dModule(3, 8, 2, activation_callable=activation_callable)

    activation_callable = partial(nn.Softmax)
    with pytest.raises(ValueError, match="Unsupported activation"):
        # softmax is not supported
        Conv2dModule(3, 8, 2, activation_callable=activation_callable)

    # conv + norm + act
    conv = Conv2dModule(3, 8, 2, norm_cfg={"type": "BN"})
    assert conv.with_activation
    assert isinstance(conv.activation, nn.Module)
    assert conv.with_norm
    assert hasattr(conv, "norm_layer")
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv + act
    conv = Conv2dModule(3, 8, 2)
    assert conv.with_activation
    assert isinstance(conv.activation, nn.Module)
    assert not conv.with_norm
    assert conv.norm_layer is None
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv
    conv = Conv2dModule(3, 8, 2, activation_callable=None)
    assert not conv.with_norm
    assert conv.norm_layer is None
    assert not conv.with_activation
    assert conv.activation is None
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    conv = Conv2dModule(3, 8, 3, padding=1, with_spectral_norm=True)
    assert hasattr(conv.conv, "weight_orig")
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    conv = Conv2dModule(3, 8, 3, padding=1, padding_mode="reflect")
    assert isinstance(conv.padding_layer, nn.ReflectionPad2d)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # non-existing padding mode
    with pytest.raises(KeyError):
        conv = Conv2dModule(3, 8, 3, padding=1, padding_mode="non_exists")

    # leaky relu
    conv = Conv2dModule(3, 8, 3, padding=1, activation_callable=nn.LeakyReLU)
    assert isinstance(conv.activation, nn.LeakyReLU)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # tanh
    conv = Conv2dModule(3, 8, 3, padding=1, activation_callable=nn.Tanh)
    assert isinstance(conv.activation, nn.Tanh)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # Sigmoid
    conv = Conv2dModule(3, 8, 3, padding=1, activation_callable=nn.Sigmoid)
    assert isinstance(conv.activation, nn.Sigmoid)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # PReLU
    conv = Conv2dModule(3, 8, 3, padding=1, activation_callable=nn.PReLU)
    assert isinstance(conv.activation, nn.PReLU)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # Test norm layer with name
    conv = Conv2dModule(3, 8, 2, norm_cfg={"type": "BN", "name": "some_norm_layer"})
    assert conv.norm_layer.__class__.__name__ == "BatchNorm2d"
    assert conv.norm_name == "some_norm_layer"
    assert hasattr(conv, "norm_layer")
    assert hasattr(conv, "some_norm_layer")
    assert not hasattr(conv, "bn")
    assert conv.some_norm_layer == conv.norm_layer


def test_bias():
    # bias: auto, without norm
    conv = Conv2dModule(3, 8, 2)
    assert conv.conv.bias is not None

    # bias: auto, with norm
    conv = Conv2dModule(3, 8, 2, norm_cfg={"type": "BN"})
    assert conv.conv.bias is None

    # bias: False, without norm
    conv = Conv2dModule(3, 8, 2, bias=False)
    assert conv.conv.bias is None

    # bias: True, with batch norm
    with pytest.warns(UserWarning) as record:
        Conv2dModule(3, 8, 2, bias=True, norm_cfg={"type": "BN"})
    assert len(record) == 1
    assert record[0].message.args[0] == "Unnecessary conv bias before batch/instance norm"

    # bias: True, with instance norm
    with pytest.warns(UserWarning) as record:
        Conv2dModule(3, 8, 2, bias=True, norm_cfg={"type": "IN"})
    assert len(record) == 1
    assert record[0].message.args[0] == "Unnecessary conv bias before batch/instance norm"


class TestDepthwiseSeparableConvModule:
    def test_forward_with_default_config(self) -> None:
        # test default config
        conv = DepthwiseSeparableConvModule(3, 8, 2)
        assert conv.depthwise_conv.conv.groups == 3
        assert conv.pointwise_conv.conv.kernel_size == (1, 1)
        assert not conv.depthwise_conv.with_norm
        assert not conv.pointwise_conv.with_norm
        assert conv.depthwise_conv.activation.__class__.__name__ == "ReLU"
        assert conv.pointwise_conv.activation.__class__.__name__ == "ReLU"
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

    def test_forward_with_spectral_norm_padding_mode(self) -> None:
        x = torch.rand(1, 3, 256, 256)

        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, with_spectral_norm=True)
        assert hasattr(conv.depthwise_conv.conv, "weight_orig")
        assert hasattr(conv.pointwise_conv.conv, "weight_orig")
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)

        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, padding_mode="reflect")
        assert isinstance(conv.depthwise_conv.padding_layer, nn.ReflectionPad2d)
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)

    def test_forward_with_dw_activation_callable(self) -> None:
        # test dw_activation_callable
        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, dw_activation_callable=nn.LeakyReLU)
        x = torch.rand(1, 3, 256, 256)
        assert conv.depthwise_conv.activation.__class__.__name__ == "LeakyReLU"
        assert conv.pointwise_conv.activation.__class__.__name__ == "ReLU"
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)

    def test_forward_with_pw_activation_callable(self) -> None:
        # test pw_activation_callable
        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, pw_activation_callable=nn.LeakyReLU)
        x = torch.rand(1, 3, 256, 256)
        assert conv.depthwise_conv.activation.__class__.__name__ == "ReLU"
        assert conv.pointwise_conv.activation.__class__.__name__ == "LeakyReLU"
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)

    def test_forward_with_activation_callable(self) -> None:
        # test activation_callable
        conv = DepthwiseSeparableConvModule(3, 8, 3, padding=1, activation_callable=nn.LeakyReLU)
        x = torch.rand(1, 3, 256, 256)
        assert conv.depthwise_conv.activation.__class__.__name__ == "LeakyReLU"
        assert conv.pointwise_conv.activation.__class__.__name__ == "LeakyReLU"
        output = conv(x)
        assert output.shape == (1, 8, 256, 256)
