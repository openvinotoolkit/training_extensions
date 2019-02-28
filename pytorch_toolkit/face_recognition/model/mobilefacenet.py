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

import math
import torch.nn as nn

from losses.am_softmax import AngleSimpleLinear
from model.blocks.mobilenet_v2_blocks import InvertedResidual
from model.blocks.shared_blocks import make_activation
from .common import ModelInterface


def init_block(in_channels, out_channels, stride, activation=nn.PReLU):
    """Builds the first block of the MobileFaceNet"""
    return nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        make_activation(activation)
    )


class MobileFaceNet(ModelInterface):
    """Implements modified MobileFaceNet from https://arxiv.org/abs/1804.07573"""
    def __init__(self, embedding_size=128, num_classes=1, width_multiplier=1., feature=True):
        super(MobileFaceNet, self).__init__()
        assert embedding_size > 0
        assert num_classes > 0
        assert width_multiplier > 0
        self.feature = feature

        # Set up of inverted residual blocks
        inverted_residual_setting = [
            # t, c, n, s
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 1],
            [4, 128, 1, 2],
            [2, 128, 2, 1]
        ]

        first_channel_num = 64
        last_channel_num = 512
        self.features = [init_block(3, first_channel_num, 2)]

        self.features.append(nn.Conv2d(first_channel_num, first_channel_num, 3, 1, 1,
                                       groups=first_channel_num, bias=False))
        self.features.append(nn.BatchNorm2d(64))
        self.features.append(nn.PReLU())

        # Inverted Residual Blocks
        in_channel_num = first_channel_num
        size_h, size_w = MobileFaceNet.get_input_res()
        size_h, size_w = size_h // 2, size_w // 2
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                if i == 0:
                    size_h, size_w = size_h // s, size_w // s
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          s, t, outp_size=(size_h, size_w)))
                else:
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          1, t, outp_size=(size_h, size_w)))
                in_channel_num = output_channel

        # 1x1 expand block
        self.features.append(nn.Sequential(nn.Conv2d(in_channel_num, last_channel_num, 1, 1, 0, bias=False),
                                           nn.BatchNorm2d(last_channel_num),
                                           nn.PReLU()))
        self.features = nn.Sequential(*self.features)

        # Depth-wise pooling
        k_size = (MobileFaceNet.get_input_res()[0] // 16, MobileFaceNet.get_input_res()[1] // 16)
        self.dw_pool = nn.Conv2d(last_channel_num, last_channel_num, k_size,
                                 groups=last_channel_num, bias=False)
        self.dw_bn = nn.BatchNorm2d(last_channel_num)
        self.conv1_extra = nn.Conv2d(last_channel_num, embedding_size, 1, stride=1, padding=0, bias=False)

        if not self.feature:
            self.fc_angular = AngleSimpleLinear(embedding_size, num_classes)

        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.dw_bn(self.dw_pool(x))
        x = self.conv1_extra(x)

        if self.feature or not self.training:
            return x

        x = x.view(x.size(0), -1)
        y = self.fc_angular(x)

        return x, y

    @staticmethod
    def get_input_res():
        return 128, 128

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.

    def init_weights(self):
        """Initializes weights of the model before training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
