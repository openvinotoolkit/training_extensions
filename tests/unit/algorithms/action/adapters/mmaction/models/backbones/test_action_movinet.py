"""Unit test for otx.algorithms.action.adapters.mmaction.models.backbones.movinet"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch
from torch import nn

from otx.algorithms.action.adapters.mmaction.models.backbones.movinet import (
    BasicBneck,
    Conv2dBNActivation,
    ConvBlock3D,
    MoViNet,
    OTXMoViNet,
    SqueezeExcitation,
    TFAvgPool3D,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestConv2dBNActivation:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.layer = Conv2dBNActivation(3, 16, kernel_size=3, padding=1)

    @e2e_pytest_unit
    def test_conv2d_bn_activation_output_shape(self):
        x = torch.Tensor(1, 3, 32, 32)
        output = self.layer(x)
        assert output.shape == (1, 16, 32, 32)

    @e2e_pytest_unit
    def test_conv2d_bn_activation_attributes(self):
        assert self.layer.kernel_size == (3, 3)
        assert self.layer.stride == (1, 1)
        assert self.layer.out_channels == 16


class TestConvBlock3D:
    @e2e_pytest_unit
    def test_conv_block_3d_output_shape(self):
        x = torch.Tensor(1, 3, 32, 32, 32)
        layer = ConvBlock3D(3, 16, kernel_size=(3, 3, 3), tf_like=True, conv_type="3d")
        output = layer(x)
        assert output.shape == (1, 16, 32, 32, 32)

    @e2e_pytest_unit
    @pytest.mark.parametrize("conv_type", ["3d", "2plus1d"])
    def test_conv_block_3d_attributes(self, conv_type):
        layer = ConvBlock3D(3, 16, kernel_size=(3, 3, 3), tf_like=True, conv_type=conv_type)
        assert layer.kernel_size == (3, 3, 3)
        assert layer.stride == (1, 1, 1)
        assert layer.dim_pad == 2
        assert layer.conv_type == conv_type
        assert layer.tf_like


class TestSqueezeExcitation:
    @pytest.fixture
    def se_block(self):
        return SqueezeExcitation(16, nn.ReLU, nn.Sigmoid, conv_type="2plus1d", squeeze_factor=4, bias=True)

    @e2e_pytest_unit
    def test_scale_output_shape(self, se_block):
        x = torch.Tensor(1, 16, 32, 32, 32)
        scale = se_block._scale(x)
        assert scale.shape == (1, 16, 1, 1, 1)

    @e2e_pytest_unit
    def test_forward_output_shape(self, se_block):
        x = torch.Tensor(1, 16, 32, 32, 32)
        output = se_block(x)
        assert output.shape == (1, 16, 32, 32, 32)

    @e2e_pytest_unit
    def test_se_block_attributes(self, se_block):
        assert se_block.fc1.kernel_size == (1, 1, 1)
        assert se_block.fc2.kernel_size == (1, 1, 1)
        assert se_block.fc1.conv_type == "2plus1d"
        assert se_block.fc2.conv_type == "2plus1d"


class TestTFAvgPool3D:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.pool = TFAvgPool3D()

    @e2e_pytest_unit
    def test_tf_avg_pool_output_shape(self):
        x = torch.Tensor(1, 3, 32, 32, 32)
        output = self.pool(x)
        assert output.shape == (1, 3, 32, 16, 16)

    @e2e_pytest_unit
    def test_tf_avg_pool_output_shape_odd(self):
        x = torch.Tensor(1, 3, 31, 31, 31)
        output = self.pool(x)
        assert output.shape == (1, 3, 31, 16, 16)

    @e2e_pytest_unit
    def test_tf_avg_pool_output_shape_odd_padding(self):
        x = torch.Tensor(1, 3, 30, 30, 30)
        output = self.pool(x)
        assert output.shape == (1, 3, 30, 15, 15)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TestBasicBneck:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.config = AttrDict(
            input_channels=64,
            expanded_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1),
            padding_avg=(1, 1, 1),
        )

    @e2e_pytest_unit
    def test_basic_bneck_output_shape(self):
        module = BasicBneck(self.config, tf_like=False, conv_type="3d", activation_layer=nn.ReLU)
        x = torch.randn(1, self.config.input_channels, 32, 32, 32)
        output = module(x)
        assert output.shape == (1, self.config.out_channels, 32, 32, 32)


class TestMoViNet:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.cfg = AttrDict()
        self.cfg.conv1 = AttrDict(
            {
                "input_channels": 3,
                "out_channels": 16,
                "kernel_size": (3, 5, 5),
                "stride": (1, 1, 1),
                "padding": (1, 2, 2),
            }
        )
        self.cfg.blocks = [
            [
                AttrDict(
                    {
                        "input_channels": 16,
                        "expanded_channels": 24,
                        "out_channels": 24,
                        "kernel_size": (3, 3, 3),
                        "stride": (1, 1, 1),
                        "padding": (1, 1, 1),
                    }
                ),
            ]
        ]
        self.cfg.conv7 = AttrDict(
            {
                "input_channels": 40,
                "out_channels": 256,
                "kernel_size": (1, 1, 1),
                "stride": (1, 1, 1),
                "padding": (0, 0, 0),
            }
        )

    @e2e_pytest_unit
    def test_movinet_output_shape(self):
        module = MoViNet(self.cfg)
        x = torch.randn(1, 3, 32, 32, 32)
        module.conv1 = nn.Identity()
        module.blocks = nn.Identity()
        module.conv7 = nn.Identity()
        output = module(x)
        assert output.shape == (1, 3, 1, 1, 1)

    @e2e_pytest_unit
    def test_init_weights(self):
        module = MoViNet(self.cfg)
        module.apply(module._init_weights)
        for m in module.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    assert m.bias.mean().item() == pytest.approx(0, abs=1e-2)
                    assert m.bias.std().item() == pytest.approx(0, abs=1e-2)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
                assert m.bias.mean().item() == pytest.approx(0, abs=1e-2)
                assert m.bias.std().item() == pytest.approx(0, abs=1e-2)
            elif isinstance(m, nn.Linear):
                assert m.bias.mean().item() == pytest.approx(0, abs=1e-2)
                assert m.bias.std().item() == pytest.approx(0, abs=1e-2)

    @e2e_pytest_unit
    def test_OTXMoViNet(self):
        model = OTXMoViNet()
        input_tensor = torch.randn(1, 3, 32, 224, 224)
        output_tensor = model(input_tensor)
        assert output_tensor.shape == torch.Size([1, 480, 1, 1, 1])
