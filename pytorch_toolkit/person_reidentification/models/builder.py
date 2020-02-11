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

from .osnet_fpn import *

from torchreid.models import __model_factory


__model_factory['fpn_osnet_x1_0'] = fpn_osnet_x1_0
__model_factory['fpn_osnet_x0_75'] = fpn_osnet_x0_75
__model_factory['fpn_osnet_x0_5'] = fpn_osnet_x0_5
__model_factory['fpn_osnet_x0_25'] = fpn_osnet_x0_25
__model_factory['fpn_osnet_ibn_x1_0'] = fpn_osnet_ibn_x1_0


def build_model(name, num_classes, loss='softmax', pretrained=True,
                use_gpu=True, dropout_cfg=None, feature_dim=512, fpn_cfg=None,
                pooling_type='avg', input_size=(256, 128), IN_first=False,
                extra_blocks=False):
    """A function wrapper for building a model.
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        dropout_cfg=dropout_cfg,
        feature_dim=feature_dim,
        fpn_cfg=fpn_cfg,
        pooling_type=pooling_type,
        input_size=input_size,
        IN_first=IN_first,
        extra_blocks=extra_blocks
    )
