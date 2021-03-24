"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 The initial implementation is taken from https://github.com/ZitongYu/CDCN (MIT License)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0):

        super().__init__()
        self.theta = theta
        self.bias = bias or None
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        if self.groups > 1:
            self.weight = nn.Parameter(kaiming_init(out_channels, in_channels//in_channels, kernel_size))
        else:
            self.weight = nn.Parameter(kaiming_init(out_channels, in_channels, kernel_size))
        self.padding = padding
        self.i = 0

    def forward(self, x):
        out_normal = F.conv2d(input=x, weight=self.weight, bias=self.bias, dilation=self.dilation,
                              stride=self.stride, padding=self.padding, groups=self.groups)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.weight.sum(dim=(2,3), keepdim=True)
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.bias, dilation=self.dilation,
                                stride=self.stride, padding=0, groups=self.groups)
            return out_normal - self.theta * out_diff

def kaiming_init(c_out, c_in, k):
    return torch.randn(c_out, c_in, k, k)*math.sqrt(2./c_in)


class Dropout(nn.Module):
    DISTRIBUTIONS = ['bernoulli', 'gaussian', 'none']

    def __init__(self, p=0.5, mu=0.5, sigma=0.3, dist='bernoulli', linear=False):
        super().__init__()

        self.dist = dist
        assert self.dist in Dropout.DISTRIBUTIONS

        self.p = float(p)
        assert 0. <= self.p <= 1.

        self.mu = float(mu)
        self.sigma = float(sigma)
        assert self.sigma > 0.
        # need to distinct 2d and 1d dropout
        self.linear = linear
    def forward(self, x):
        if self.dist == 'bernoulli' and not self.linear:
            out = F.dropout2d(x, self.p, self.training)
        elif self.dist == 'bernoulli' and self.linear:
            out = F.dropout(x, self.p, self.training)
        elif self.dist == 'gaussian':
            if self.training:
                with torch.no_grad():
                    soft_mask = x.new_empty(x.size()).normal_(self.mu, self.sigma).clamp_(0., 1.)

                scale = 1. / self.mu
                out = scale * soft_mask * x
            else:
                out = x
        else:
            out = x

        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_in(inp, oup, stride, theta):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 3, stride, 1, bias=False, theta=theta),
        nn.InstanceNorm2d(oup),
        h_swish()
    )

def conv_3x3_bn(inp, oup, stride, theta):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 3, stride, 1, bias=False, theta=theta),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_in(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.InstanceNorm2d(oup),
        h_swish()
    )

def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNet(nn.Module):
    """parent class for mobilenets"""
    def __init__(self, width_mult, prob_dropout, type_dropout,
                 prob_dropout_linear, embeding_dim, mu, sigma,
                 theta, scaling, multi_heads):
        super().__init__()
        self.prob_dropout = prob_dropout
        self.type_dropout = type_dropout
        self.width_mult = width_mult
        self.prob_dropout_linear = prob_dropout_linear
        self.embeding_dim = embeding_dim
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.scaling = scaling
        self.multi_heads = multi_heads
        self.features = nn.Identity

        # building last several layers
        self.conv_last = nn.Identity
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.spoofer = nn.Linear(embeding_dim, 2)
        if self.multi_heads:
            self.lightning = nn.Linear(embeding_dim, 5)
            self.spoof_type = nn.Linear(embeding_dim, 11)
            self.real_atr = nn.Linear(embeding_dim, 40)

    def forward(self, x):
        x = self.features(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        return x

    def make_logits(self, features, all=False):
        all = all if self.multi_heads else False
        output = features.view(features.size(0), -1)
        spoof_out = self.spoofer(output)
        if all:
            type_spoof = self.spoof_type(output)
            lightning_type = self.lightning(output)
            real_atr = torch.sigmoid(self.real_atr(output))
            return spoof_out, type_spoof, lightning_type, real_atr
        return spoof_out

    def forward_to_onnx(self,x):
        x = self.features(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        spoof_out = self.spoofer(x)
        if isinstance(spoof_out, tuple):
            spoof_out = spoof_out[0]
        probab = F.softmax(spoof_out*self.scaling, dim=-1)
        return probab
