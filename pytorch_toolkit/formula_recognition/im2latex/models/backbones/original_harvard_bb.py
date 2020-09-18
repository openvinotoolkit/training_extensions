"""
 Copyright (c) 2020 Intel Corporation

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


class Im2LatexBackBone(nn.Module):
    def __init__(self):
        super().__init__()
        # follow the original paper's table2: CNN specification
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2), (0, 0)),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (1, 1))
        )

    def forward(self, imgs):
        return self.cnn_encoder(imgs)
