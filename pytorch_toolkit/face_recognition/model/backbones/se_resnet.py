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

from model.blocks.se_resnet_blocks import SEBottleneck


class SEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, activation=nn.ReLU):
        self.inplanes = 64
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], activation=activation)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, activation=activation)
        self.avgpool = nn.Conv2d(512 * block.expansion, 512 * block.expansion, 7,
                                 groups=512 * block.expansion, bias=False)
        self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, activation=nn.ReLU):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def se_resnet50(**kwargs):
    model = SEResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet101(**kwargs):
    model = SEResNet(SEBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def se_resnet152(**kwargs):
    model = SEResNet(SEBottleneck, [3, 8, 36, 3], **kwargs)
    return model
