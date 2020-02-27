import collections
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ...utils import drop_last

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def check_conv1_params(model, pretrained_weights):
    if model.conv1.in_channels != pretrained_weights['conv1.weight'].size(1):
        # get mean over RGB channels weights
        rgb_mean = torch.mean(pretrained_weights['conv1.weight'], dim=1)

        expand_ratio = model.conv1.in_channels // pretrained_weights['conv1.weight'].size(1)
        pretrained_weights['conv1.weight'] = pretrained_weights['conv1.weight'].repeat(1, expand_ratio, 1, 1)
        # pretrained_weights['conv1.weight'] = rgb_mean.unsqueeze(1).repeat(1, model.conv1.in_channels, 1, 1)


def average_conv1_weights(old_params, in_channels):
    new_params = collections.OrderedDict()
    layer_count = 0
    all_key_list = old_params.keys()
    for layer_key in drop_last(all_key_list, 2):
        if layer_count == 0:
            rgb_weight = old_params[layer_key]
            rgb_weight_mean = torch.mean(rgb_weight, dim=1)
            flow_weight = rgb_weight_mean.unsqueeze(1).repeat(1, in_channels, 1, 1)
            if isinstance(flow_weight, torch.autograd.Variable):
                new_params[layer_key] = flow_weight.data
            else:
                new_params[layer_key] = flow_weight
            layer_count += 1
        else:
            new_params[layer_key] = old_params[layer_key]
            layer_count += 1

    return new_params


def load_pretrained_resnet(model, resnet_name='resnet34', num_channels=3):
    if num_channels == 3:
        pretrained_weights = model_zoo.load_url(model_urls[resnet_name])
        check_conv1_params(model, pretrained_weights)
        model.load_state_dict(pretrained_weights)
    else:
        pretrained_dict = model_zoo.load_url(model_urls[resnet_name])
        model_dict = model.state_dict()

        new_pretrained_dict = average_conv1_weights(pretrained_dict, num_channels)

        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet18', num_channels)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet34', num_channels)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet50', num_channels)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet101', num_channels)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet152', num_channels)
    return model
