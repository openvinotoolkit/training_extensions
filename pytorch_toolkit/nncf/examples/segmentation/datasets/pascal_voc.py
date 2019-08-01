"""
 Copyright (c) 2019 Intel Corporation
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
from torchvision.datasets import VOCSegmentation


class PascalVoc(VOCSegmentation):
    color_encoding = OrderedDict([
        ('background', (0, 0, 0)),
        ('aeroplane', (128, 0, 0)),
        ('bicycle', (0, 128, 0)),
        ('bird', (128, 128, 0)),
        ('boat', (0, 0, 128)),
        ('bottle', (128, 0, 128)),
        ('bus', (0, 128, 128)),
        ('car', (128, 128, 128)),
        ('cat', (64, 0, 0)),
        ('chair', (192, 0, 0)),
        ('cow', (64, 128, 0)),
        ('diningtable', (192, 128, 0)),
        ('dog', (64, 0, 128)),
        ('horse', (192, 0, 128)),
        ('motorbike', (64, 128, 128)),
        ('person', (192, 128, 128)),
        ('pottedplant', (0, 64, 0)),
        ('sheep', (128, 64, 0)),
        ('sofa', (0, 192, 0)),
        ('train', (128, 192, 0)),
        ('tvmonitor', (0, 64, 128))
    ])

    VOID_LABEL = 255

    #  __init__ is unchanged from torchvision.datasets.VOCSegmentation

    def __getitem__(self, index):
        img, label = super(PascalVoc, self).__getitem__(index)

        # This affects mIoU computations. Do not uncomment
        # the line below if you want the same mIoU as in the
        # torchvision reference scripts.

        # label[label == self.VOID_LABEL] = 0
        return img, label
