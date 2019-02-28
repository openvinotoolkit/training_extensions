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

from .common import ModelInterface


class LandmarksNet(ModelInterface):
    """Facial landmarks localization network"""
    def __init__(self):
        super(LandmarksNet, self).__init__()
        self.bn_first = nn.BatchNorm2d(3)
        activation = nn.PReLU
        self.landnet = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            activation(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation(),
            nn.BatchNorm2d(128)
        )
        # dw pooling
        self.bottleneck_size = 256
        self.pool = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=6, padding=0, groups=128),
            activation(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, self.bottleneck_size, kernel_size=1, padding=0),
            activation(),
            nn.BatchNorm2d(self.bottleneck_size),
        )
        # Regressor for 5 landmarks (10 coordinates)
        self.fc_loc = nn.Sequential(
            nn.Conv2d(self.bottleneck_size, 64, kernel_size=1),
            activation(),
            nn.Conv2d(64, 10, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        xs = self.landnet(self.bn_first(x))
        xs = self.pool(xs)
        xs = self.fc_loc(xs)
        return xs

    def get_input_res(self):
        return 48, 48

    def set_dropout_ratio(self, ratio):
        pass
