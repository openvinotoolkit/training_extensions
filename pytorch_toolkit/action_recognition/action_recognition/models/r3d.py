"""
This is an re-implementation of R3D and R(2+1) topologies from paper:

Tran, Du, et al. "A Closer Look at Spatiotemporal Convolutions for Action Recognition."
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
"""
import collections
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from action_recognition.utils import drop_last, get_fine_tuning_parameters

__all__ = ['R3D', 'R3D_18', 'R3D_34', 'R3D_101', 'R3D_152', 'R2p1D_18', 'R2p1D_34', 'R2p1D_101', 'R2p1D_152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

R3D_MODELS = {
    'r3d': lambda args, encoder: R3D_18(
        num_classes=args.n_classes
    ),
    'r2+1d': lambda args, encoder: R2p1D_18(
        num_classes=args.n_classes,
    ),

    'r3d_18': lambda args, encoder: R3D_18(
        num_classes=args.n_classes
    ),
    'r2+1d_18': lambda args, encoder: R2p1D_18(
        num_classes=args.n_classes,
    ),
    'r3d_34': lambda args, encoder: R3D_34(
        num_classes=args.n_classes
    ),
    'r2+1d_34': lambda args, encoder: R2p1D_34(
        num_classes=args.n_classes,
    ),
    'r3d_50': lambda args, encoder: R3D_50(
        num_classes=args.n_classes
    ),
    'r2+1d_50': lambda args, encoder: R2p1D_50(
        num_classes=args.n_classes,
    ),
    'r3d_101': lambda args, encoder: R3D_101(
        num_classes=args.n_classes
    ),
    'r2+1d_101': lambda args, encoder: R2p1D_101(
        num_classes=args.n_classes,
    ),
    'r3d_152': lambda args, encoder: R3D_152(
        num_classes=args.n_classes
    ),
    'r2+1d_152': lambda args, encoder: R2p1D_152(
        num_classes=args.n_classes,
    ),
}


def make_conv(in_planes, out_planes, middle_planes=None, kernel_size=(3, 3, 3), stride=(1, 1, 1), decomposed=True,
              bias=False):
    padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
    if decomposed:
        i = 3 * in_planes * out_planes * kernel_size[1] * kernel_size[2]
        i /= in_planes * kernel_size[1] * kernel_size[2] + 3 * out_planes
        if middle_planes is None:
            middle_planes = int(i)
        return nn.Sequential(
            nn.Conv3d(in_planes, middle_planes, kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=(0, padding[1], padding[2]), stride=stride, bias=bias),
            nn.BatchNorm3d(middle_planes),
            nn.ReLU(),
            nn.Conv3d(middle_planes, out_planes, kernel_size=(kernel_size[0], 1, 1),
                      padding=(padding[0], 0, 0), stride=1, bias=bias)
        )
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class BasicBlockR3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, decomposed=True):
        super(BasicBlockR3D, self).__init__()
        self.conv1 = make_conv(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, decomposed=decomposed)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = make_conv(planes, planes, kernel_size=(3, 3, 3), decomposed=decomposed)
        self.bn2 = nn.BatchNorm3d(planes)
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


class BottleneckR3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, decomposed=True):
        super(BottleneckR3D, self).__init__()
        self.conv1 = make_conv(inplanes, planes, kernel_size=(1, 1, 1), decomposed=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = make_conv(planes, planes, kernel_size=(3, 3, 3), stride=stride, decomposed=decomposed)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = make_conv(planes, planes * 4, kernel_size=(1, 1, 1), decomposed=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class R3D(nn.Module):

    def __init__(self, block, layers, num_classes=400, num_channels=3, decomposed=True):
        self.inplanes = 64
        super(R3D, self).__init__()

        self.decomposed = decomposed

        self.conv1 = make_conv(num_channels, 64, middle_planes=45, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                               decomposed=decomposed)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], downsample=True)
        self.layer3 = self._make_layer(block, 256, layers[2], downsample=True)
        self.layer4 = self._make_layer(block, 512, layers[3], downsample=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, downsample=False):
        downsample_layer = None
        first_stride = (2, 2, 2) if downsample else (1, 1, 1)
        if downsample or self.inplanes != planes * block.expansion:
            downsample_layer = nn.Sequential(
                make_conv(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=first_stride,
                          decomposed=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, first_stride, downsample_layer, decomposed=self.decomposed)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, decomposed=self.decomposed))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)


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


def R3D_18(pretrained=False, **kwargs):
    """Constructs a R3D-18 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = R3D(BasicBlockR3D, [2, 2, 2, 2], decomposed=False, **kwargs)

    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet18', num_channels)
    return model


def R3D_34(pretrained=False, **kwargs):
    """Constructs a R3D-34 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = R3D(BasicBlockR3D, [3, 4, 6, 3], decomposed=False, **kwargs)
    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet34', num_channels)
    return model


def R2p1D_18(**kwargs):
    """Constructs a R2+1D-34 model."""
    model = R3D(BasicBlockR3D, [2, 2, 2, 2], decomposed=True, **kwargs)
    return model


def R2p1D_34(**kwargs):
    """Constructs a R2+1D-34 model."""
    model = R3D(BasicBlockR3D, [3, 4, 6, 3], decomposed=True, **kwargs)
    return model


def R3D_50(pretrained=False, **kwargs):
    """Constructs a R3D-50 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = R3D(BottleneckR3D, [3, 4, 6, 3], decomposed=False, **kwargs)
    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet50', num_channels)
    return model


def R3D_101(pretrained=False, **kwargs):
    """Constructs a R3D-101 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = R3D(BottleneckR3D, [3, 4, 23, 3], decomposed=False, **kwargs)
    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet101', num_channels)
    return model


def R3D_152(pretrained=False, **kwargs):
    """Constructs a R3D-152 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = R3D(BottleneckR3D, [3, 8, 36, 3], decomposed=False, **kwargs)
    num_channels = 3
    if 'num_channels' in kwargs:
        num_channels = kwargs['num_channels']
    if pretrained:
        model = load_pretrained_resnet(model, 'resnet152', num_channels)
    return model


def R2p1D_50(**kwargs):
    """Constructs a R2+1D-50 model."""
    model = R3D(BottleneckR3D, [3, 4, 6, 3], decomposed=True, **kwargs)
    return model


def R2p1D_101(**kwargs):
    """Constructs a R2+1D-101 model."""
    model = R3D(BottleneckR3D, [3, 4, 23, 3], decomposed=True, **kwargs)
    return model


def R2p1D_152(**kwargs):
    """Constructs a R2+1D-152 model."""
    model = R3D(BottleneckR3D, [3, 8, 36, 3], decomposed=True, **kwargs)
    return model
