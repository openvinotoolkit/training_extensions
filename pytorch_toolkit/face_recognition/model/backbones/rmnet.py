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

from collections import OrderedDict

import torch.nn as nn
from ..blocks.rmnet_blocks import RMBlock


class RMNetBody(nn.Module):
    def __init__(self, block=RMBlock, blocks_per_stage=(None, 4, 8, 10, 11), trunk_width=(32, 32, 64, 128, 256),
                 bottleneck_width=(None, 8, 16, 32, 64)):
        super(RMNetBody, self).__init__()
        assert len(blocks_per_stage) == len(trunk_width) == len(bottleneck_width)
        self.dim_out = trunk_width[-1]

        stages = [nn.Sequential(OrderedDict([
            ('data_bn', nn.BatchNorm2d(3)),
            ('conv1', nn.Conv2d(3, trunk_width[0], kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(trunk_width[0])),
            ('relu1', nn.ReLU(inplace=True))])), ]

        for i, (blocks_num, w, wb) in enumerate(zip(blocks_per_stage, trunk_width, bottleneck_width)):
            # Zeroth stage is already added.
            if i == 0:
                continue
            stage = []
            # Do not downscale input to the first stage.
            if i > 1:
                stage.append(block(trunk_width[i - 1], wb, w, downsample=True))
            for _ in range(blocks_num):
                stage.append(block(w, wb, w))
            stages.append(nn.Sequential(*stage))

        self.stages = nn.Sequential(OrderedDict([('stage_{}'.format(i), stage) for i, stage in enumerate(stages)]))

        self.init_weights()

    def init_weights(self):
        m = self.stages[0][0]  # ['data_bn']
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        m = self.stages[0][1]  # ['conv1']
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        m = self.stages[0][2]  # ['bn1']
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        # All other blocks should be initialized internally during instantiation.

    def forward(self, x):
        return self.stages(x)
