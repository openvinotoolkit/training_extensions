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

import os

import torch
import torch.nn as nn
from examples.object_detection.layers import L2Norm
from examples.object_detection.layers.modules.ssd_head import MultiOutputSequential, SSDDetectionOutput
from nncf.checkpoint_loading import load_state

from examples.common.example_logger import logger


BASE_NUM_OUTPUTS = {
    300: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    512: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
}
EXTRAS_NUM_OUTPUTS = {
    300: [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    512: [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'K', 256],
}

BASE_OUTPUT_INDICES = {
    300: [12],
    512: [12],
}

EXTRA_OUTPUT_INDICES = {
    300: [2, 5, 7, 9],
    512: [2, 5, 8, 11, 14],
}


class SSD_VGG(nn.Module):
    def __init__(self, cfg, size, num_classes, batch_norm=False):
        super(SSD_VGG, self).__init__()
        self.config = cfg
        self.num_classes = num_classes
        self.size = size
        self.enable_batchmorm = batch_norm

        base_layers, base_outs, base_feats = build_vgg_ssd_layers(
            BASE_NUM_OUTPUTS[size], BASE_OUTPUT_INDICES[size], batch_norm=batch_norm
        )
        extra_layers, extra_outs, extra_feats = build_vgg_ssd_extra(
            EXTRAS_NUM_OUTPUTS[size], EXTRA_OUTPUT_INDICES[size], batch_norm=batch_norm
        )
        self.basenet = MultiOutputSequential(base_outs, base_layers)
        self.extras = MultiOutputSequential(extra_outs, extra_layers)

        self.detection_head = SSDDetectionOutput(base_feats + extra_feats, num_classes, cfg)
        self.L2Norm = L2Norm(512, 20, 1e-10)

    def forward(self, x):
        img_tensor = x[0].clone().unsqueeze(0)

        sources, x = self.basenet(x)
        sources[0] = self.L2Norm(sources[0])

        extra_sources, x = self.extras(x)

        return self.detection_head(sources + extra_sources, img_tensor)

    def load_weights(self, base_file):
        _, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            logger.debug('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            logger.debug('Finished!')
        else:
            logger.error('Sorry only .pth and .pkl files supported.')


def make_ssd_vgg_layer(input_features, output_features, kernel=3, padding=1, dilation=1, modifier=None,
                       batch_norm=False):
    stride = 1
    if modifier == 'S':
        stride = 2
        padding = 1
    elif modifier == 'K':
        kernel = 4
        padding = 1

    layer = [nn.Conv2d(input_features, output_features, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation)]
    if batch_norm:
        layer.append(nn.BatchNorm2d(output_features))
    layer.append(nn.ReLU(inplace=True))
    return layer


def build_vgg_ssd_layers(num_outputs, output_inddices, start_input_channels=3, batch_norm=False):
    vgg_layers = []
    output_num_features = []
    source_indices = []
    in_planes = start_input_channels
    modifier = None
    for i, out_planes in enumerate(num_outputs):
        if out_planes in ('M', 'C'):
            vgg_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=1 if modifier == 'C' else 0))
            continue
        if isinstance(out_planes, str):
            modifier = out_planes
            continue
        vgg_layers.extend(make_ssd_vgg_layer(in_planes, out_planes, modifier=modifier, batch_norm=batch_norm))
        modifier = None
        in_planes = out_planes
        if i in output_inddices:
            source_indices.append(len(vgg_layers) - 1)
            output_num_features.append(out_planes)

    vgg_layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
    vgg_layers.extend(make_ssd_vgg_layer(in_planes, 1024, kernel=3, padding=6, dilation=6, batch_norm=batch_norm))
    vgg_layers.extend(make_ssd_vgg_layer(1024, 1024, kernel=1, batch_norm=batch_norm))

    source_indices.append(len(vgg_layers) - 1)
    output_num_features.append(1024)
    return vgg_layers, source_indices, output_num_features


def build_vgg_ssd_extra(num_outputs, output_indices, statrt_input_channels=1024, batch_norm=False):
    extra_layers = []
    output_num_features = []
    source_indices = []
    in_planes = statrt_input_channels
    modifier = None
    kernel_sizes = (1, 3)
    for i, out_planes in enumerate(num_outputs):
        if isinstance(out_planes, str):
            modifier = out_planes
            continue
        kernel = kernel_sizes[len(extra_layers) % 2]
        extra_layers.extend(make_ssd_vgg_layer(in_planes, out_planes, modifier=modifier, kernel=kernel, padding=0,
                                               batch_norm=batch_norm))
        modifier = None
        in_planes = out_planes
        if i in output_indices:
            source_indices.append(len(extra_layers) - 1)
            output_num_features.append(out_planes)

    return extra_layers, source_indices, output_num_features


def build_ssd_vgg(cfg, size, num_classes, config):
    ssd_vgg = SSD_VGG(cfg, size, num_classes, batch_norm=config.get('batchnorm', False))

    if config.basenet and (config.resuming_checkpoint is None) and (config.weights is None):
        logger.debug('Loading base network...')
        basenet_weights = torch.load(config.basenet)
        new_weights = {}
        for wn, wv in basenet_weights.items():
            wn = wn.replace('features.', '')
            new_weights[wn] = wv

        load_state(ssd_vgg.basenet, new_weights, is_resume=False)
    return ssd_vgg
