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
    def __init__(self, arch, disable_layer_3, disable_layer_4, output_channels=512,
                 enable_last_conv=False, one_ch_first_conv=False, check_num_out_channels=True):
        super().__init__()
        self.output_channels = output_channels
        self.arch = arch
        _resnet = architectures.get(arch, None)
        if _resnet is None:
            raise ValueError("Unkown backbone, please, check the field 'arch' in the backbone_config")
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
            if check_num_out_channels:
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
        return x


class CustomResNetLikeBackbone(ResNetLikeBackbone):
    def __init__(self, arch, disable_layer_3, disable_layer_4, output_channels, enable_last_conv, one_ch_first_conv, custom_parameters):
        super().__init__(arch, disable_layer_3, disable_layer_4, output_channels=output_channels,
                         enable_last_conv=enable_last_conv, one_ch_first_conv=one_ch_first_conv,
                         check_num_out_channels=False)
        self.inplanes = custom_parameters['inplanes']
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        conv1_params = custom_parameters['conv1']
        _resnet = architectures.get(arch, None)(pretrained=False, progress=True)
        _resnet.inplanes = self.inplanes
        self.conv1 = nn.Conv2d(1 if one_ch_first_conv else 3,
                               self.inplanes,
                               conv1_params['kernel'],
                               conv1_params['stride'],
                               conv1_params['padding'],
                               bias=self.conv1.bias
                               )
        self.return_layers = custom_parameters['return_layers']
        layers = custom_parameters['layers']
        layer_strides = custom_parameters['layer_strides']
        planes = custom_parameters['planes']
        block = torchvision.models.resnet.Bottleneck
        self.use_maxpool = custom_parameters['use_maxpool']
        self.layer1 = _resnet._make_layer(block, planes[0], layers[0], stride=layer_strides[0])
        self.layer2 = _resnet._make_layer(block, planes[1], layers[1], stride=layer_strides[1])
        self.layer3 = _resnet._make_layer(block, planes[2], layers[2], stride=layer_strides[2])
        self.layer4 = _resnet._make_layer(block, planes[3], layers[3], stride=layer_strides[3])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_maxpool:
            x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        if self.layer3 is None:
            return x2
        x3 = self.layer3(x2)
        if self.layer4 is None:
            return x3
        x4 = self.layer4(x3)

        if not self.return_layers:
            return x4
        out = []
        if 2 in self.return_layers:
            out.append(x2)
        if 3 in self.return_layers:
            out.append(x3)
        if 4 in self.return_layers:
            out.append(x4)
        return out


class ResNetLikeWithSkipsBetweenLayers(CustomResNetLikeBackbone):
    def __init__(self, arch, disable_layer_3, disable_layer_4, output_channels, enable_last_conv, one_ch_first_conv, custom_parameters):
        super().__init__(arch, disable_layer_3, disable_layer_4, output_channels,
                         enable_last_conv, one_ch_first_conv, custom_parameters)
        layer_strides = custom_parameters['layer_strides']
        self.connect_1_3 = nn.Conv2d(
            in_channels=self.layer1[-1].conv1.out_channels,
            out_channels=self.layer3[-1].conv1.out_channels,
            kernel_size=1,
            stride=(layer_strides[1]-1)*2 + (layer_strides[2]-1)*2,
            padding=0)
        self.connect_2_4 = nn.Conv2d(
            in_channels=self.layer2[-1].conv1.out_channels,
            out_channels=self.layer4[-1].conv1.out_channels,
            kernel_size=1,
            stride=(layer_strides[2]-1)*2 + (layer_strides[3]-1)*2,
            padding=0)
        super()._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_maxpool:
            x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        if self.layer3 is None:
            return x2

        x3 = self.layer3(x2) + self.connect_1_3(x1)

        if self.layer4 is None:
            return x3

        x4 = self.layer4(x3) + self.connect_2_4(x2)

        return x4
