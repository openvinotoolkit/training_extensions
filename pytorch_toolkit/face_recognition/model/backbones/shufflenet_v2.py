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

from model.blocks.shufflenet_v2_blocks import ShuffleInvertedResidual, conv_bn, conv_1x1_bn


class ShuffleNetV2Body(nn.Module):
    def __init__(self, input_size=224, width_mult=1.):
        super(ShuffleNetV2Body, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError("Unsupported width multiplier")

        # building first layer
        self.bn_first = nn.BatchNorm2d(3)
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)

        self.features = []

        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleInvertedResidual(input_channel, output_channel,
                                                                 2, 2, activation=nn.PReLU))
                else:
                    self.features.append(ShuffleInvertedResidual(input_channel, output_channel,
                                                                 1, 1, activation=nn.PReLU))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1], activation=nn.PReLU)
        self.init_weights()

    @staticmethod
    def get_downscale_factor():
        return 16

    def init_weights(self):
        m = self.bn_first
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    def get_num_output_channels(self):
        return self.stage_out_channels[-1]

    def forward(self, x):
        x = self.conv1(self.bn_first(x))
        x = self.features(x)
        x = self.conv_last(x)
        return x
