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

from losses.am_softmax import AngleSimpleLinear
from model.backbones.rmnet import RMNetBody
from model.blocks.rmnet_blocks import RMBlock
from .common import ModelInterface


class RMNetAngular(ModelInterface):
    """Face reid head for the ResMobNet architecture. See https://arxiv.org/pdf/1812.02465.pdf for details
    about the ResMobNet backbone."""
    def __init__(self, embedding_size, num_classes=0, feature=True, body=RMNetBody):
        super(RMNetAngular, self).__init__()
        self.feature = feature
        self.backbone = body()
        self.global_pooling = nn.MaxPool2d((8, 8))
        self.conv1_extra = nn.Conv2d(256, embedding_size, 1, stride=1, padding=0, bias=False)
        if not feature:
            self.fc_angular = AngleSimpleLinear(embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pooling(x)
        x = self.conv1_extra(x)

        if self.feature or not self.training:
            return x

        x = x.view(x.size(0), -1)
        y = self.fc_angular(x)

        return x, y

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.

        for m in self.backbone.modules():
            if isinstance(m, RMBlock):
                m.dropout_ratio = ratio

    @staticmethod
    def get_input_res():
        return 128, 128
