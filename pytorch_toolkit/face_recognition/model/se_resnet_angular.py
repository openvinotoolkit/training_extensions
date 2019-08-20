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

from losses.am_softmax import AngleSimpleLinear
from model.backbones.se_resnet import se_resnet50
from .common import ModelInterface


class SEResNetAngular(ModelInterface):
    """Face reid head for the SE ResNet architecture"""
    def __init__(self, embedding_size=128, num_classes=0, feature=True, loss=None, base=se_resnet50):
        super(SEResNetAngular, self).__init__()

        self.bn_first = nn.BatchNorm2d(3)
        self.feature = feature
        self.model = base(num_classes=embedding_size, activation=nn.PReLU)
        self.feat_bn = nn.BatchNorm2d(self.model.get_output_channels())
        self.conv1_extra = nn.Conv2d(self.model.get_output_channels(), embedding_size,
                                     1, stride=1, padding=0, bias=False)
        self.emb_bn = nn.BatchNorm2d(embedding_size)
        self.embedding_size = embedding_size
        self.dropout_ratio = 0.4
        self.loss = loss
        if not self.feature:
            self.fc_angular = AngleSimpleLinear(self.embedding_size, num_classes)

    def forward(self, x, target=None):
        assert self.loss is not None or not self.training
        x = self.bn_first(x)
        x = self.feat_bn(self.model(x))
        if self.dropout_ratio > 0:
            x = F.dropout(x, p=self.dropout_ratio, training=self.training, inplace=True)
        x = self.emb_bn(self.conv1_extra(x))

        if self.feature or not self.training:
            return x.view(x.shape[0], x.shape[1], 1, 1)

        x = x.view(x.size(0), -1)
        y = self.fc_angular(x)

        return x, y, self.loss(y, target)

    @staticmethod
    def get_input_res():
        return 112, 112

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.
        self.dropout_ratio = ratio
