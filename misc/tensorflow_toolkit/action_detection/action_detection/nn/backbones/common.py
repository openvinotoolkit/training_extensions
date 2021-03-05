# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from action_detection.nn.backbones.rmnet import RMNet
from action_detection.nn.backbones.twinnet import TwinNet


def get_backbone(backbone_name, net_input, fn_activation, is_training, merge_bn, merge_bn_transition,
                 name=None, use_extra_layers=False, keep_probe=0.9, norm_kernels=False):
    """Returns backbone according specified parameters.

    :param backbone_name: Name of backbone
    :param net_input: Network input
    :param fn_activation: Main activation function
    :param is_training: Training indicator variable
    :param merge_bn: Whether to run with merged BatchNorms
    :param merge_bn_transition: Whether to run in BatchNorm merging mode
    :param name: Name of node
    :param use_extra_layers: Whether to include extra layers if available
    :param keep_probe: Probability to keep value in dropout
    :param norm_kernels: Whether to L2 normalize convolution weights
    :return: Parameterised backbone
    """

    if backbone_name == 'rmnet':
        backbone = RMNet(net_input, fn_activation, is_training, merge_bn, merge_bn_transition, name,
                         use_extra_layers, keep_probe, norm_kernels)
    elif backbone_name == 'twinnet':
        backbone = TwinNet(net_input, fn_activation, is_training, merge_bn, merge_bn_transition, name,
                           keep_probe, norm_kernels)
    else:
        raise Exception('Unknown backbone name: {}'.format(backbone_name))

    return backbone


def get_orthogonal_scope_name(backbone_name):
    """Returns scope name of convolutions for orthogonal regularization.

    :param backbone_name: Name of backbone
    :return: Name of scope
    """

    if backbone_name == 'rmnet' or backbone_name == 'twinnet':
        return 'dim_red'
    elif backbone_name == 'shufflenetv2':
        return 'inner_map'
    else:
        raise Exception('Unknown backbone name: {}'.format(backbone_name))
