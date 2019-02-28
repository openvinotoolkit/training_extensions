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

from model.blocks.shared_blocks import SELayer


class InvertedResidual(nn.Module):
    """Implementation of the modified Inverted residual block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, outp_size=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.inv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.PReLU(),

            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, 1,
                      groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.PReLU(),

            nn.Conv2d(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            SELayer(out_channels, 8, nn.PReLU, outp_size)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_block(x)

        return self.inv_block(x)
