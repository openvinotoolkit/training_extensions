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

import logging

from text_spotting.models.backbones.efficientnet import EfficientNet
from text_spotting.models.backbones.mobilenet_v2 import MobileNetV2
from text_spotting.models.backbones.mobilenet_v3 import mobilenetv3_large
from text_spotting.models.backbones.resnet import resnet50


def get_backbone(name, **kwargs):
    if name == 'mobilenet_v2':
        backbone = MobileNetV2()
        # backbone.freeze_stages_params(range(2))
        # backbone.freeze_stages_bns(range(19))
        backbone.set_output_stages((3, 6, 13, 18))
    elif name == 'resnet50':
        replace_stride_with_dilation = None
        if 'replace_stride_with_dilation' in kwargs:
            replace_stride_with_dilation = kwargs['replace_stride_with_dilation']

        backbone = resnet50(pretrained=True, progress=True,
                            replace_stride_with_dilation=replace_stride_with_dilation)
        output_stages = (1, 2, 3, 4)

    elif name == 'mobilenet_v3_large':
        backbone = mobilenetv3_large(shape=kwargs['shape'])
        # backbone.freeze_stages_params(range(2))
        # backbone.freeze_stages_bns(range(16))
        backbone.set_output_stages((3, 6, 12, 15))
    elif name == 'efficientnet_b0':
        backbone = EfficientNet.from_name('efficientnet-b0', shape=kwargs['shape'])
        backbone.freeze_stages_params(range(2))
        backbone.freeze_stages_bns(range(17))
        backbone.set_output_stages((3, 5, 11, 16))
    else:
        raise IOError(f'Invalid backbone name {name}')

    if 'freeze_stages_params' in kwargs:
        logging.info(f'Freezing stages params {kwargs["freeze_stages_params"]}')
        backbone.freeze_stages_params(range(kwargs['freeze_stages_params']))
    if 'freeze_stages_bns' in kwargs:
        logging.info(f'Freezing stages bns {kwargs["freeze_stages_bns"]}')
        backbone.freeze_stages_bns(range(kwargs['freeze_stages_bns']))

    if 'output_stages' in kwargs:
        output_stages = kwargs['output_stages']
    logging.info(f'Setting output stages {output_stages}')
    backbone.set_output_stages(output_stages)

    return backbone
