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

import torch.nn as nn
import torch.nn.functional as F

from model.blocks.shared_blocks import make_activation


class RMBlock(nn.Module):
    def __init__(self, input_planes, squeeze_planes, output_planes, downsample=False, dropout_ratio=0.1,
                 activation=nn.ELU):
        super(RMBlock, self).__init__()
        self.downsample = downsample
        self.input_planes = input_planes
        self.output_planes = output_planes

        self.squeeze_conv = nn.Conv2d(input_planes, squeeze_planes, kernel_size=1, bias=False)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)

        self.dw_conv = nn.Conv2d(squeeze_planes, squeeze_planes, groups=squeeze_planes, kernel_size=3, padding=1,
                                 stride=2 if downsample else 1, bias=False)
        self.dw_bn = nn.BatchNorm2d(squeeze_planes)

        self.expand_conv = nn.Conv2d(squeeze_planes, output_planes, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(output_planes)

        self.activation = make_activation(activation)

        self.dropout_ratio = dropout_ratio

        if self.downsample:
            self.skip_conv = nn.Conv2d(input_planes, output_planes, kernel_size=1, bias=False)
            self.skip_conv_bn = nn.BatchNorm2d(output_planes)

        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.activation(self.squeeze_bn(self.squeeze_conv(x)))
        out = self.activation(self.dw_bn(self.dw_conv(out)))
        out = self.expand_bn(self.expand_conv(out))
        if self.dropout_ratio > 0:
            out = F.dropout(out, p=self.dropout_ratio, training=self.training, inplace=True)
        if self.downsample:
            residual = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
            residual = self.skip_conv(residual)
            residual = self.skip_conv_bn(residual)
        out += residual
        return self.activation(out)
