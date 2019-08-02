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

from examples.object_detection.models.ssd_mobilenet import build_ssd_mobilenet
from examples.object_detection.models.ssd_vgg import build_ssd_vgg


def build_ssd(net_name, cfg, ssd_dim, num_classes, config):
    assert net_name in ['ssd_vgg', 'ssd_mobilenet']
    if net_name == 'ssd_vgg':
        model = build_ssd_vgg(cfg, ssd_dim, num_classes, config)
    if net_name == 'ssd_mobilenet':
        model = build_ssd_mobilenet(cfg, ssd_dim, num_classes, config)

    return model
