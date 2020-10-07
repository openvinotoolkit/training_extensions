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
            kernel_diff = self.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.bias, dilation=self.dilation,
                                stride=self.stride, padding=0, groups=self.groups)
            return out_normal - self.theta * out_diff

def kaiming_init(c_out, c_in, k):
    return torch.randn(c_out, c_in, k, k)*math.sqrt(2./c_in)
