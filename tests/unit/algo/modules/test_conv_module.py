# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmcv/blob/main/tests/test_cnn/test_conv_module.py
from unittest.mock import patch

import pytest
import torch
from otx.algo.modules.conv_module import ConvModule
from torch import nn


def test_conv_module():
    conv_cfg = "conv"
    with pytest.raises(AssertionError):
        # conv_cfg must be a dict or None
        ConvModule(3, 8, 2, conv_cfg=conv_cfg)

    norm_cfg = "norm"
    with pytest.raises(AssertionError):
        # norm_cfg must be a dict or None
        ConvModule(3, 8, 2, norm_cfg=norm_cfg)

    act_cfg = {"type": "softmax"}
    with pytest.raises(KeyError):
        # softmax is not supported
        ConvModule(3, 8, 2, act_cfg=act_cfg)

    # conv + norm + act
    conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN"})
    assert conv.with_activation
    assert hasattr(conv, "activate")
    assert conv.with_norm
    assert hasattr(conv, "norm_layer")
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv + norm with efficient mode
    efficient_conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN"}, efficient_conv_bn_eval=True).eval()
    plain_conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN"}, efficient_conv_bn_eval=False).eval()
    for efficient_param, plain_param in zip(efficient_conv.state_dict().values(), plain_conv.state_dict().values()):
        plain_param.copy_(efficient_param)

    efficient_mode_output = efficient_conv(x)
    plain_mode_output = plain_conv(x)
    assert torch.allclose(efficient_mode_output, plain_mode_output, atol=1e-5)

    # `conv` attribute can be dynamically modified in efficient mode
    efficient_conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN"}, efficient_conv_bn_eval=True).eval()
    new_conv = nn.Conv2d(3, 8, 2).eval()
    efficient_conv.conv = new_conv
    efficient_mode_output = efficient_conv(x)
    plain_mode_output = efficient_conv.activate(efficient_conv.norm_layer(new_conv(x)))
    assert torch.allclose(efficient_mode_output, plain_mode_output, atol=1e-5)

    # conv + act
    conv = ConvModule(3, 8, 2)
    assert conv.with_activation
    assert hasattr(conv, "activate")
    assert not conv.with_norm
    assert conv.norm_layer is None
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv
    conv = ConvModule(3, 8, 2, act_cfg=None)
    assert not conv.with_norm
    assert conv.norm_layer is None
    assert not conv.with_activation
    assert not hasattr(conv, "activate")
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    conv = ConvModule(3, 8, 3, padding=1, with_spectral_norm=True)
    assert hasattr(conv.conv, "weight_orig")
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    conv = ConvModule(3, 8, 3, padding=1, padding_mode="reflect")
    assert isinstance(conv.padding_layer, nn.ReflectionPad2d)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # non-existing padding mode
    with pytest.raises(KeyError):
        conv = ConvModule(3, 8, 3, padding=1, padding_mode="non_exists")

    # leaky relu
    conv = ConvModule(3, 8, 3, padding=1, act_cfg={"type": "LeakyReLU"})
    assert isinstance(conv.activate, nn.LeakyReLU)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # tanh
    conv = ConvModule(3, 8, 3, padding=1, act_cfg={"type": "Tanh"})
    assert isinstance(conv.activate, nn.Tanh)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # Sigmoid
    conv = ConvModule(3, 8, 3, padding=1, act_cfg={"type": "Sigmoid"})
    assert isinstance(conv.activate, nn.Sigmoid)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # PReLU
    conv = ConvModule(3, 8, 3, padding=1, act_cfg={"type": "PReLU"})
    assert isinstance(conv.activate, nn.PReLU)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # Test norm layer with name
    conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN", "name": "some_norm_layer"})
    assert conv.norm_layer.__class__.__name__ == "BatchNorm2d"
    assert conv.norm_name == "some_norm_layer"
    assert hasattr(conv, "norm_layer")
    assert hasattr(conv, "some_norm_layer")
    assert not hasattr(conv, "bn")
    assert conv.some_norm_layer == conv.norm_layer


def test_bias():
    # bias: auto, without norm
    conv = ConvModule(3, 8, 2)
    assert conv.conv.bias is not None

    # bias: auto, with norm
    conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN"})
    assert conv.conv.bias is None

    # bias: False, without norm
    conv = ConvModule(3, 8, 2, bias=False)
    assert conv.conv.bias is None

    # bias: True, with batch norm
    with pytest.warns(UserWarning) as record:
        ConvModule(3, 8, 2, bias=True, norm_cfg={"type": "BN"})
    assert len(record) == 1
    assert record[0].message.args[0] == "Unnecessary conv bias before batch/instance norm"

    # bias: True, with instance norm
    with pytest.warns(UserWarning) as record:
        ConvModule(3, 8, 2, bias=True, norm_cfg={"type": "IN"})
    assert len(record) == 1
    assert record[0].message.args[0] == "Unnecessary conv bias before batch/instance norm"


def conv_forward(self, x):
    return x + "_conv"


def bn_forward(self, x):
    return x + "_bn"


def relu_forward(self, x):
    return x + "_relu"


@patch("torch.nn.ReLU.forward", relu_forward)
@patch("torch.nn.BatchNorm2d.forward", bn_forward)
@patch("torch.nn.Conv2d.forward", conv_forward)
def test_order():
    order = ["conv", "norm", "act"]
    with pytest.raises(AssertionError):
        # order must be a tuple
        ConvModule(3, 8, 2, order=order)

    order = ("conv", "norm")
    with pytest.raises(AssertionError):
        # length of order must be 3
        ConvModule(3, 8, 2, order=order)

    order = ("conv", "norm", "norm")
    with pytest.raises(AssertionError):
        # order must be an order of 'conv', 'norm', 'act'
        ConvModule(3, 8, 2, order=order)

    order = ("conv", "norm", "something")
    with pytest.raises(AssertionError):
        # order must be an order of 'conv', 'norm', 'act'
        ConvModule(3, 8, 2, order=order)

    conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN"})
    out = conv("input")
    assert out == "input_conv_bn_relu"

    conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN"}, order=("norm", "conv", "act"))
    out = conv("input")
    assert out == "input_bn_conv_relu"

    conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN"})
    out = conv("input", activate=False)
    assert out == "input_conv_bn"

    conv = ConvModule(3, 8, 2, norm_cfg={"type": "BN"})
    out = conv("input", norm=False)
    assert out == "input_conv_relu"
