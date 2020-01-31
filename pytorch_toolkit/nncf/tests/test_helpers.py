"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np
import torch
from torch import nn

from nncf.config import Config


def fill_conv_weight(conv, value):
    conv.weight.data.fill_(value)
    with torch.no_grad():
        mask = torch.eye(conv.kernel_size[0])
        conv.weight += mask


def fill_bias(module, value):
    module.bias.data.fill_(value)


def fill_linear_weight(linear, value):
    linear.weight.data.fill_(value)
    with torch.no_grad():
        n = min(linear.in_features, linear.out_features)
        linear.weight[:n, :n] += torch.eye(n)


def create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    fill_conv_weight(conv, weight_init)
    fill_bias(conv, bias_init)
    return conv


class BasicConvTestModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, kernel_size=2, weight_init=-1, bias_init=-2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.conv = create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init)

    @staticmethod
    def default_weight():
        return torch.tensor([[[[0., -1.],
                               [-1., 0.]]], [[[0., -1.],
                                              [-1., 0.]]]])

    @staticmethod
    def default_bias():
        return torch.tensor([-2., -2])

    def forward(self, x):
        return self.conv(x)

    @property
    def weights_num(self):
        return self.out_channels * self.kernel_size ** 2

    @property
    def bias_num(self):
        return self.kernel_size

    @property
    def nz_weights_num(self):
        return self.kernel_size * self.out_channels

    @property
    def nz_bias_num(self):
        return self.kernel_size


def test_basic_model_has_expected_params():
    model = BasicConvTestModel()
    act_weights = model.conv.weight.data
    ref_weights = BasicConvTestModel.default_weight()
    act_bias = model.conv.bias.data
    ref_bias = BasicConvTestModel.default_bias()

    check_equal(act_bias, ref_bias)
    check_equal(act_weights, ref_weights)

    assert act_weights.nonzero().size(0) == model.nz_weights_num
    assert act_bias.nonzero().size(0) == model.nz_bias_num
    assert act_weights.numel() == model.weights_num
    assert act_bias.numel() == model.bias_num


def test_basic_model_is_valid():
    model = BasicConvTestModel()
    input_ = torch.ones([1, 1, 4, 4])
    ref_output = torch.ones((1, 2, 3, 3)) * (-4)
    act_output = model(input_)
    check_equal(ref_output, act_output)


class TwoConvTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []
        self.features.append(nn.Sequential(create_conv(1, 2, 2, -1, -2)))
        self.features.append(nn.Sequential(create_conv(2, 1, 3, 0, 0)))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)

    @property
    def weights_num(self):
        return 8 + 18

    @property
    def bias_num(self):
        return 2 + 1

    @property
    def nz_weights_num(self):
        return 4 + 6

    @property
    def nz_bias_num(self):
        return 2


def test_two_conv_model_has_expected_params():
    model = TwoConvTestModel()
    act_weights_1 = model.features[0][0].weight.data
    act_weights_2 = model.features[1][0].weight.data
    act_bias_1 = model.features[0][0].bias.data
    act_bias_2 = model.features[1][0].bias.data

    ref_weights_1 = BasicConvTestModel.default_weight()
    channel = torch.eye(3, 3).reshape([1, 1, 3, 3])
    ref_weights_2 = torch.cat((channel, channel), 1)

    check_equal(act_weights_1, ref_weights_1)
    check_equal(act_weights_2, ref_weights_2)

    check_equal(act_bias_1, BasicConvTestModel.default_bias())
    check_equal(act_bias_2, torch.tensor([0]))

    assert act_weights_1.nonzero().size(0) + act_weights_2.nonzero().size(0) == model.nz_weights_num
    assert act_bias_1.nonzero().size(0) + act_bias_2.nonzero().size(0) == model.nz_bias_num
    assert act_weights_1.numel() + act_weights_2.numel() == model.weights_num
    assert act_bias_1.numel() + act_bias_2.numel() == model.bias_num


def test_two_conv_model_is_valid():
    model = TwoConvTestModel()
    input_ = torch.ones([1, 1, 4, 4])
    ref_output = torch.tensor([-24])
    act_output = model(input_)
    check_equal(ref_output, act_output)


def get_empty_config(model_size=4, input_sample_size=(1, 1, 4, 4)):
    config = Config()
    config.update({
        "model": "basic_sparse_conv",
        "model_size": model_size,
        "input_info":
            {
                "sample_size": input_sample_size,
            },
    })
    return config


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_equal(test, reference, rtol=1e-4):
    for i, (x, y) in enumerate(zip(test, reference)):
        y = y.cpu().detach().numpy()
        np.testing.assert_allclose(x, y, rtol=rtol, err_msg="Index: {}".format(i))
