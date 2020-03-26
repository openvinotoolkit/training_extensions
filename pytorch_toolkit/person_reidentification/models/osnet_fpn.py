"""
 MIT License

 Copyright (c) 2018 Kaiyang Zhou

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

from __future__ import absolute_import
from __future__ import division

import logging as log

import torch
from torch import nn
from torch.nn import functional as F

from torchreid.models.osnet import OSNet, ConvLayer, LightConv3x3, Conv1x1Linear, \
                                   ChannelGate, Conv1x1, pretrained_urls
from .modules.fpn import FPN
from .modules.dropout import Dropout
from .modules.gmp import GeneralizedMeanPooling


__all__ = ['fpn_osnet_x1_0', 'fpn_osnet_x0_75', 'fpn_osnet_x0_5', 'fpn_osnet_x0_25', 'fpn_osnet_ibn_x1_0']

pretrained_urls_fpn = {
    'fpn_osnet_x1_0': pretrained_urls['osnet_x1_0'],
    'fpn_osnet_x0_75': pretrained_urls['osnet_x0_75'],
    'fpn_osnet_x0_5': pretrained_urls['osnet_x0_5'],
    'fpn_osnet_x0_25': pretrained_urls['osnet_x0_25'],
    'fpn_osnet_ibn_x1_0': pretrained_urls['osnet_ibn_x1_0']
}


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4,
                 dropout_cfg=None, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)
        self.dropout = None
        if dropout_cfg is not None:
            self.dropout = Dropout(**dropout_cfg)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.dropout:
            x3 = self.dropout(x3)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNetFPN(OSNet):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
    """

    def __init__(self, num_classes, blocks, layers, channels,
                 feature_dim=256,
                 loss='softmax',
                 instance_norm=False,
                 dropout_cfg=None,
                 fpn_cfg=None,
                 pooling_type='avg',
                 input_size=(256, 128),
                 IN_first=False,
                 extra_blocks=False,
                 **kwargs):
        self.dropout_cfg = dropout_cfg
        self.extra_blocks = extra_blocks
        if self.extra_blocks:
            for i, l in enumerate(layers):
                layers[i] = l + 1
        super(OSNetFPN, self).__init__(num_classes, blocks, layers, channels, feature_dim, loss, instance_norm)

        self.feature_scales = (4, 8, 16, 16)
        if fpn_cfg is not None:
            self.fpn_enable = fpn_cfg.enable
            self.fpn_dim = fpn_cfg.dim
            self.fpn_process = fpn_cfg.process
            assert self.fpn_process in ['concatenation', 'max_pooling', 'elementwise_sum']
        else:
            self.fpn_enable = False
        self.feature_dim = feature_dim

        self.use_IN_first = IN_first
        if IN_first:
            self.in_first = nn.InstanceNorm2d(3, affine=True)
            self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=self.use_IN_first)

        if self.fpn_enable:
            self.fpn = FPN(channels, self.feature_scales, self.fpn_dim, self.fpn_dim)
            fpn_out_dim = self.fpn_dim if self.fpn_process in ['max_pooling', 'elementwise_sum'] \
                          else feature_dim
            self.fc = self._construct_fc_layer(feature_dim, fpn_out_dim, dropout_cfg)
        else:
            self.fpn = None
            self.fc = self._construct_fc_layer(feature_dim, channels[3], dropout_cfg)

        if self.loss not in ['am_softmax', ]:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            from engine.losses.am_softmax import AngleSimpleLinear
            self.classifier = AngleSimpleLinear(self.feature_dim, num_classes)

        if 'conv' in pooling_type:
            kernel_size = (input_size[0] // self.feature_scales[-1], input_size[1] // self.feature_scales[-1])
            if self.fpn_enable:
                self.global_avgpool = nn.Conv2d(fpn_out_dim, fpn_out_dim, kernel_size, groups=fpn_out_dim)
            else:
                self.global_avgpool = nn.Conv2d(channels[3], channels[3], kernel_size, groups=channels[3])
        elif 'avg' in pooling_type:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        elif 'gmp' in pooling_type:
            self.global_avgpool = GeneralizedMeanPooling()
        else:
            raise ValueError('Incorrect pooling type')

        if self.fpn_enable and self.fpn_process == 'concatenation':
            self.fpn_extra_conv = ConvLayer(self.fpn_dim * len(self.fpn.dims_out),
                                            feature_dim, 3, stride=1, padding=1, IN=False)
        else:
            self.fpn_extra_conv = None

        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN, dropout_cfg=self.dropout_cfg))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            if self.loss not in ['am_softmax', ]:
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.PReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p.p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def featuremaps(self, x):
        out = []
        if self.use_IN_first:
            x = self.in_first(x)
        x = self.conv1(x)
        x1 = self.maxpool(x)
        out.append(x1)
        x2 = self.conv2(x1)
        out.append(x2)
        x3 = self.conv3(x2)
        out.append(x3)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        out.append(x5)
        return x5, out

    def forward(self, x, return_featuremaps=False, get_embeddings=False):
        x, feature_pyramid = self.featuremaps(x)
        if self.fpn is not None:
            feature_pyramid = self.fpn(feature_pyramid)
            x = self.process_feature_pyramid(feature_pyramid)

        if return_featuremaps:
            return x

        v = self.global_avgpool(x)
        if isinstance(self.fc[0], nn.Linear):
            v = v.view(v.size(0), -1)

        if self.fc is not None:
            if self.training:
                v = self.fc(v)
            else:
                v = self.fc[0](v).view(v.size(0), -1, 1)
                v = self.fc[1](v)
                v = self.fc[2](v)
        v = v.view(v.size(0), -1)

        if not self.training:
            return v

        y = self.classifier(v)

        if get_embeddings:
            return v, y

        if self.loss in ['softmax', 'adacos', 'd_softmax', 'am_softmax']:
            return y
        elif self.loss in ['triplet', ]:
            return v, y
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

    def process_feature_pyramid(self, feature_pyramid):
        feature_pyramid = feature_pyramid[:-1]
        target_shape = feature_pyramid[-1].shape[2:]
        for i in range(len(feature_pyramid) - 1):
            kernel_size = int(feature_pyramid[i].shape[2] // target_shape[0])
            feature_pyramid[i] = nn.functional.max_pool2d(feature_pyramid[i], kernel_size=kernel_size)
            if self.fpn_process == 'max_pooling':
                feature_pyramid[-1] = torch.max(feature_pyramid[i], feature_pyramid[-1])
            elif self.fpn_process == 'elementwise_sum':
                feature_pyramid[-1] = torch.add(feature_pyramid[i], feature_pyramid[-1])
            else:
                feature_pyramid[-1] = torch.cat((feature_pyramid[i], feature_pyramid[-1]), dim=1)
        if self.fpn_process == 'concatenation':
            output = self.fpn_extra_conv(feature_pyramid[-1])
        else:
            output = feature_pyramid[-1]
        return output


def fpn_osnet_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNetFPN(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                     channels=[64, 256, 384, 512], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='fpn_osnet_x1_0')
    return model


def fpn_osnet_x0_75(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNetFPN(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                     channels=[48, 192, 288, 384], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='fpn_osnet_x0_75')
    return model


def fpn_osnet_x0_5(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNetFPN(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                     channels=[32, 128, 192, 256], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='fpn_osnet_x0_5')
    return model


def fpn_osnet_x0_25(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNetFPN(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                     channels=[16, 64, 96, 128], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='fpn_osnet_x0_25')
    return model


def fpn_osnet_ibn_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # standard size (width x1.0) + IBN layer
    model = OSNetFPN(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                     channels=[64, 256, 384, 512], loss=loss, IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='fpn_osnet_ibn_x1_0')
    return model


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(ENV_TORCH_HOME,
                      os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls_fpn[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        log.warning(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(cached_file))
    else:
        print('Successfully loaded imagenet pretrained weights from "{}"'.format(cached_file))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                  'due to unmatched keys or layer size: {}'.format(discarded_layers))
