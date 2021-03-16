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
import torchvision.models

architectures = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
    'resnet101': torchvision.models.resnet101,
    'resnet152': torchvision.models.resnet152,
    'resnext50_32x4d': torchvision.models.resnext50_32x4d,
    'resnext101_32x8d': torchvision.models.resnext101_32x8d,
}


class ResNetLikeBackbone(nn.Module):
    def __init__(self, arch, disable_layer_3, disable_layer_4, output_channels=512, enable_last_conv=False, one_ch_first_conv=False):
        super(ResNetLikeBackbone, self).__init__()
        self.output_channels = output_channels
        self.arch = arch
        _resnet = architectures.get(arch, None)
        if _resnet is None:
            raise ValueError('Unkown backbone, please, check the field 'arch' in the backbone_config')
        _resnet = _resnet(pretrained=True, progress=True)
        self.groups = _resnet.groups
        self.base_width = _resnet.base_width
        conv_1 = _resnet.conv1
        self.conv1 = nn.Conv2d(1, conv_1.out_channels, conv_1.kernel_size,
                               conv_1.stride, conv_1.padding, bias=conv_1.bias) if one_ch_first_conv else conv_1
        self.bn1 = _resnet.bn1
        self.relu = _resnet.relu
        self.maxpool = _resnet.maxpool
        self.layer1 = _resnet.layer1
        self.layer2 = _resnet.layer2
        enable_layer_3 = not disable_layer_3
        enable_layer_4 = not disable_layer_4
        if arch in ('resnet18', 'resnet34'):
            out_ch = 128
        else:
            out_ch = 512
        if enable_layer_4:
            assert enable_layer_3, 'Cannot enable layer4 w/out enabling layer 3'

        if enable_layer_3 and disable_layer_4:
            self.layer3 = _resnet.layer3
            self.layer4 = None
            if arch in ('resnet18', 'resnet34'):
                out_ch = 256
            else:
                out_ch = 1024
        elif enable_layer_3 and enable_layer_4:
            self.layer3 = _resnet.layer3
            self.layer4 = _resnet.layer4
            if arch in ('resnet18', 'resnet34'):
                out_ch = 512
            else:
                out_ch = 2048
        else:
            self.layer3 = None
            self.layer4 = None
        print('Initialized cnn encoder {}'.format(arch))
        if enable_last_conv:
            print('Last conv enabled')
            self.last_conv = nn.Conv2d(out_ch, self.output_channels, 1)
        else:
            self.last_conv = None
            assert out_ch == self.output_channels, f"""
            Number of the output channels ({out_ch}) of the backbone from the config should be equal
            to actual number of the output channels ({self.output_channels})
            """

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        if self.last_conv is not None:
            x = self.last_conv(x)
        return x
