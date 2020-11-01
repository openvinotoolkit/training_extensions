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

from .model_tools import *

class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride,
                 use_se, use_hs, prob_dropout, type_dropout, sigma, mu):
        super().__init__()
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


class MobileNetV3(MobileNet):
    def __init__(self, cfgs, mode, **kwargs):
        super().__init__(**kwargs)
        self.cfgs = cfgs
        # setting of inverted residual blocks
        assert mode in ['large', 'small']
        # building first layer
        input_channel = make_divisible(16 * self.width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, theta=self.theta)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = make_divisible(c * self.width_mult, 8)
            exp_size = make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs,
                                                                prob_dropout=self.prob_dropout,
                                                                mu=self.mu,
                                                                sigma=self.sigma,
                                                                type_dropout=self.type_dropout))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.conv_last = conv_1x1_bn(input_channel, self.embeding_dim)

        self.spoofer = nn.Sequential(
            Dropout(p=self.prob_dropout_linear,
                    mu=self.mu,
                    sigma=self.sigma,
                    dist=self.type_dropout,
                    linear=True),
            nn.BatchNorm1d(self.embeding_dim),
            h_swish(),
            nn.Linear(self.embeding_dim, 2),
        )
        if self.multi_heads:
            self.lightning = nn.Sequential(
                Dropout(p=self.prob_dropout_linear,
                        mu=self.mu,
                        sigma=self.sigma,
                        dist=self.type_dropout,
                        linear=True),
                nn.BatchNorm1d(self.embeding_dim),
                h_swish(),
                nn.Linear(self.embeding_dim, 5),
            )
            self.spoof_type = nn.Sequential(
                Dropout(p=self.prob_dropout_linear,
                        mu=self.mu,
                        sigma=self.sigma,
                        dist=self.type_dropout,
                        linear=True),
                nn.BatchNorm1d(self.embeding_dim),
                h_swish(),
                nn.Linear(self.embeding_dim, 11),
            )
            self.real_atr = nn.Sequential(
                Dropout(p=self.prob_dropout_linear,
                        mu=self.mu,
                        sigma=self.sigma,
                        dist=self.type_dropout,
                        linear=True),
                nn.BatchNorm1d(self.embeding_dim),
                h_swish(),
                nn.Linear(self.embeding_dim, 40),
            )


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
