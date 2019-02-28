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
from model.backbones.shufflenet_v2 import ShuffleNetV2Body
from .common import ModelInterface


class ShuffleNetV2Angular(ModelInterface):
    """Face reid head for the ShuffleNetV2 architecture"""
    def __init__(self, embedding_size, num_classes=0, feature=True, body=ShuffleNetV2Body, **kwargs):
        super(ShuffleNetV2Angular, self).__init__()
        self.feature = feature
        kwargs['input_size'] = ShuffleNetV2Angular.get_input_res()[0]
        kwargs['width_mult'] = 1.
        self.backbone = body(**kwargs)
        k_size = int(kwargs['input_size'] / self.backbone.get_downscale_factor())
        self.global_pool = nn.Conv2d(self.backbone.stage_out_channels[-1], self.backbone.stage_out_channels[-1],
                                     (k_size, k_size), groups=self.backbone.stage_out_channels[-1], bias=False)
        self.conv1_extra = nn.Conv2d(self.backbone.get_num_output_channels(), embedding_size, 1, padding=0, bias=False)
        if not feature:
            self.fc_angular = AngleSimpleLinear(embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = self.conv1_extra(x)

        if self.feature or not self.training:
            return x

        x = x.view(x.size(0), -1)
        y = self.fc_angular(x)

        return x, y

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.

    @staticmethod
    def get_input_res():
        res = 128
        return res, res
