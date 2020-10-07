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

 The initial implementation is taken from  https://github.com/d-li14/mobilenetv3.pytorch (MIT License)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2d_cd import Conv2d_cd
from .dropout import Dropout


def _make_divisible(v, divisor, min_value=None):
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
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
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


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride,
                 use_se, use_hs, prob_dropout, type_dropout, sigma, mu):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        self.dropout2d = Dropout(dist=type_dropout, mu=mu ,
                                 sigma=sigma,
                                 p=prob_dropout)
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                         (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                         (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.dropout2d(self.conv(x))
        else:
            return self.dropout2d(self.conv(x))


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, prob_dropout, type_dropout,
                 prob_dropout_linear=0.5,
                 embeding_dim=1280, mu=0.5,
                 sigma=0.3, width_mult=1.,
                 theta=0, multi_heads=True, scaling=1):
        super().__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.prob_dropout = prob_dropout
        self.type_dropout = type_dropout
        self.scaling = scaling
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.multi_heads = multi_heads
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, theta=self.theta)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs,
                                                                prob_dropout=self.prob_dropout,
                                                                mu=self.mu,
                                                                sigma=self.sigma,
                                                                type_dropout=self.type_dropout))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, embeding_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # bulding heads for multi task
        self.spoofer = nn.Sequential(
            Dropout(p=prob_dropout_linear,
                    mu=self.mu,
                    sigma=self.sigma,
                    dist=self.type_dropout,
                    linear=True),
            nn.BatchNorm1d(embeding_dim),
            h_swish(),
            nn.Linear(embeding_dim, 2),
        )
        if self.multi_heads:
            self.lightning = nn.Sequential(
                Dropout(p=prob_dropout_linear,
                        mu=self.mu,
                        sigma=self.sigma,
                        dist=self.type_dropout,
                        linear=True),
                nn.BatchNorm1d(embeding_dim),
                h_swish(),
                nn.Linear(embeding_dim, 5),
            )
            self.spoof_type = nn.Sequential(
                Dropout(p=prob_dropout_linear,
                        mu=self.mu,
                        sigma=self.sigma,
                        dist=self.type_dropout,
                        linear=True),
                nn.BatchNorm1d(embeding_dim),
                h_swish(),
                nn.Linear(embeding_dim, 11),
            )
            self.real_atr = nn.Sequential(
                Dropout(p=prob_dropout_linear,
                        mu=self.mu,
                        sigma=self.sigma,
                        dist=self.type_dropout,
                        linear=True),
                nn.BatchNorm1d(embeding_dim),
                h_swish(),
                nn.Linear(embeding_dim, 40),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.conv_last(x)
        return x

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

    def make_logits(self, features):
        output = self.avgpool(features)
        output = output.view(output.size(0), -1)
        spoof_out = self.spoofer(output)
        if self.multi_heads:
            type_spoof = self.spoof_type(output)
            lightning_type = self.lightning(output)
            real_atr = torch.sigmoid(self.real_atr(output))
            return spoof_out, type_spoof, lightning_type, real_atr
        return spoof_out

    def spoof_task(self, features):
        output = self.avgpool(features)
        output = output.view(output.size(0), -1)
        spoof_out = self.spoofer(output)
        return spoof_out

def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)

def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)
