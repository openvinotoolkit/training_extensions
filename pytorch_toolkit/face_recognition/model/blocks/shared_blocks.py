"""
 Copyright (c) 2018 Intel Corporation
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

import torch
import torch.nn as nn


def make_activation(activation):
    """Factory for activation functions"""
    if activation != nn.PReLU:
        return activation(inplace=True)

    return activation()


class SELayer(nn.Module):
    """Implementation of the Squeeze-Excitaion layer from https://arxiv.org/abs/1709.01507"""
    def __init__(self, inplanes, squeeze_ratio=8, activation=nn.PReLU, size=None):
        super(SELayer, self).__init__()
        assert squeeze_ratio >= 1
        assert inplanes > 0
        if size is not None:
            self.global_avgpool = nn.AvgPool2d(size)
        else:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / squeeze_ratio), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / squeeze_ratio), inplanes, kernel_size=1, stride=1)
        self.relu = make_activation(activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out


class ScaleFilter(nn.Module):
    """Implementaion of the ScaleFilter regularizer"""
    def __init__(self, q):
        super(ScaleFilter, self).__init__()
        assert 0 < q < 1
        self.q = q

    def forward(self, x):
        if not self.training:
            return x

        scale_factors = 1. + self.q \
                - 2*self.q*torch.rand(x.shape[1], 1, 1, dtype=torch.float32, requires_grad=False).to(x.device)
        return x * scale_factors
