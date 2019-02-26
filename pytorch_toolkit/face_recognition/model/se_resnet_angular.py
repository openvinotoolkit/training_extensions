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
from model.backbones.se_resnet import se_resnet50
from .common import ModelInterface


class SEResNetAngular(ModelInterface):
    """Face reid head for the SE ResNet architecture"""
    def __init__(self, embedding_size=128, num_classes=0, feature=True):
        super(SEResNetAngular, self).__init__()

        self.bn_first = nn.BatchNorm2d(3)
        self.feature = feature
        self.model = se_resnet50(num_classes=embedding_size, activation=nn.PReLU)
        self.embedding_size = embedding_size

        if not self.feature:
            self.fc_angular = AngleSimpleLinear(self.embedding_size, num_classes)

    def forward(self, x):
        x = self.bn_first(x)
        x = self.model(x)

        if self.feature or not self.training:
            return x

        x = x.view(x.size(0), -1)
        y = self.fc_angular(x)

        return x, y

    @staticmethod
    def get_input_res():
        return 112, 112

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.
