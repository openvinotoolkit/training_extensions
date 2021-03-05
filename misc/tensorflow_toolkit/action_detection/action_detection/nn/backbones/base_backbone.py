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

from abc import ABCMeta

from action_detection.nn.nodes.initializers import orthogonal_initializer


class BaseBackbone(object):
    """Base class for backbones.
    """

    __metaclass__ = ABCMeta

    def __init__(self, net_input, fn_activation, is_training, merge_bn, merge_bn_transition, name,
                 use_extra_layers, keep_prob, reduced, norm_kernels):
        """Constructor.

        :param net_input: Network input.
        :param fn_activation: Main activation function
        :param is_training: Training indicator variable
        :param merge_bn: Whether to run with merged BatchNorms
        :param merge_bn_transition: Whether to run in BatchNorm merging mode
        :param name: Name of output network
        :param use_extra_layers: Whether to include extra layers if available
        :param keep_prob: Probability to keep value in dropout
        :param reduced: Whether to construct lightweight network variant
        :param norm_kernels: Whether to normalize convolution kernels
        """

        self._is_training = is_training
        self._merge_bn = merge_bn
        self._merge_bn_transition = merge_bn_transition
        self._fn_activation = fn_activation
        self._reduced = reduced
        self._norm_kernels = norm_kernels

        self._model = {'input': net_input}

        self._ort_init = orthogonal_initializer(mode='conv', out_scale=0.1)
        self._build(net_input, use_extra_layers, name, keep_prob)

    def _build(self, input_value, use_extra_layers, name, keep_prob):
        """Constructs target network.

        :param input_value: Network input
        :param use_extra_layers: Whether to enable extra layers
        :param name: Name of output network
        :param keep_prob: Probability to keep value in dropout
        """

    @property
    def input(self):
        """Returns network input.

        :return: Network input
        """

        return self._model['input']

    @property
    def output(self):
        """Returns network output.

        :return: Network output
        """

        return self._model['output']

    @property
    def skeleton(self):
        """Returns network skeleton.

        :return: Network skeleton
        """

        return self._model
