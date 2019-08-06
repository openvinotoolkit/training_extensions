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

import tensorflow as tf

from action_detection.nn.backbones.rmnet import RMNet
from action_detection.nn.nodes.ops import conv2d


class TwinNet(RMNet):
    """Base class for TwinNet architecture.
    """

    def __init__(self, net_input, fn_activation, is_training, merge_bn, merge_bn_transition,
                 name='twinnet', keep_prob=0.9, norm_kernels=False):
        """Constructor.

        :param net_input: Network input
        :param fn_activation: Main activation function
        :param is_training: Training indicator variable
        :param merge_bn: Whether to run with merged BatchNorms
        :param merge_bn_transition: Whether to run in BatchNorm merging mode
        :param name: Name of output network
        :param keep_prob: Probability to keep value in dropout
        :param norm_kernels: Whether to normalize convolution kernels
        """

        super(TwinNet, self).__init__(net_input, fn_activation, is_training, merge_bn, merge_bn_transition,
                                      name, False, keep_prob, False, norm_kernels)

    def _bridge(self, source, target, num_channels, name):
        """Constructs bridge between two input streams.

        :param source: Source stream input
        :param target: Target strean input
        :param num_channels: Number of input and output channels
        :param name: Name of block
        :return: Output of merged streams
        """

        with tf.variable_scope(name):
            out = conv2d(source, [1, 1, num_channels[0], num_channels[1]], 'mix',
                         is_training=self._is_training, use_bias=False, use_bn=True,
                         merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                         norm_kernel=self._norm_kernels)
            out = tf.add(out, target)
            out = self._fn_activation(out)
        return out

    def _twin_bottleneck(self, left_x, right_x, num_channels, name, keep_prob=None, k=1, bridge=False):
        """Constructs two stream bottlenecks.

        :param left_x: Left stream input
        :param right_x: Right stream input
        :param num_channels: Number of input and output channels
        :param name: Name of block
        :param keep_prob: Probability to keep value in dropout
        :param k: Stride of node
        :param bridge: Whether to enable bridge between streams
        :return: Tuple of bottleneck outputs
        """

        with tf.variable_scope('detection'):
            left_y = self._bottleneck(left_x, num_channels, name, keep_prob=keep_prob, k=k)

        with tf.variable_scope('classification'):
            right_y = self._bottleneck(right_x, num_channels, name, keep_prob=keep_prob, k=k)

        if bridge:
            with tf.variable_scope(name + '/bridge'):
                mixed_left_y = self._bridge(right_y, left_y, [num_channels[1], num_channels[1]], 'left_bridge')
                mixed_right_y = self._bridge(left_y, right_y, [num_channels[1], num_channels[1]], 'right_bridge')
                left_y, right_y = mixed_left_y, mixed_right_y

        return left_y, right_y

    def _build(self, input_value, use_extra_layers, name, keep_prob):
        """Constructs target network.

        :param input_value: Network input
        :param use_extra_layers: Whether to enable extra layers
        :param name: Name of output network
        :param keep_prob: Probability to keep value in dropout
        """

        with tf.variable_scope(name):
            tf.add_to_collection('activation_summary', tf.summary.histogram('data_stat', input_value))

            with tf.variable_scope('shared'):
                local_y = self._normalize_inputs(input_value, 3)
                tf.add_to_collection('activation_summary', tf.summary.histogram('input_stat', local_y))

                # Init block: x1 -> x1/2
                local_y = self._init_block(local_y, [3, 32], 'init_block')
                self._model['output_init'] = local_y
                tf.add_to_collection('activation_summary', tf.summary.histogram('init_block_stat', local_y))

                # Bottleneck1: x1/2 -> x1/2
                local_y = self._bottleneck(local_y, [32, 32], 'bottleneck1_1', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [32, 32], 'bottleneck1_2', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [32, 32], 'bottleneck1_3', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [32, 32], 'bottleneck1_4', keep_prob=keep_prob)
                self._model['output_2x'] = local_y
                tf.add_to_collection('activation_summary', tf.summary.histogram('output_2x_stat', local_y))

                # Bottleneck2: x1/2 -> x1/4
                local_y = self._bottleneck(local_y, [32, 64], 'bottleneck2_0', keep_prob=keep_prob, k=2)
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_1', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_2', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_3', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_4', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_5', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_6', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_7', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_8', keep_prob=keep_prob)
                shared_y = local_y
                self._model['output_4x'] = local_y
                tf.add_to_collection('activation_summary', tf.summary.histogram('output_4x_stat', local_y))

            # Bottleneck3: x1/4 -> x1/8
            det_y, cl_y = self._twin_bottleneck(shared_y, shared_y, [64, 128], 'bottleneck3_0',
                                                keep_prob=keep_prob, k=2)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_1', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_2', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_3', keep_prob=keep_prob,
                                                bridge=True)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_4', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_5', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_6', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_7', keep_prob=keep_prob,
                                                bridge=True)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_8', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_9', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 128], 'bottleneck3_10', keep_prob=keep_prob)
            self._model['det_output_8x'], self._model['cl_output_8x'] = det_y, cl_y
            tf.add_to_collection('activation_summary', tf.summary.histogram('det_output_8x_stat', det_y))
            tf.add_to_collection('activation_summary', tf.summary.histogram('cl_output_8x_stat', cl_y))

            # Bottleneck4: x1/8 -> x1/16
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [128, 256], 'bottleneck4_0', keep_prob=keep_prob, k=2)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_1', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_2', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_3', keep_prob=keep_prob,
                                                bridge=True)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_4', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_5', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_6', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_7', keep_prob=keep_prob,
                                                bridge=True)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_8', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_9', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_10', keep_prob=keep_prob)
            det_y, cl_y = self._twin_bottleneck(det_y, cl_y, [256, 256], 'bottleneck4_11', keep_prob=keep_prob)
            self._model['det_output_16x'], self._model['cl_output_16x'] = det_y, cl_y
            tf.add_to_collection('activation_summary', tf.summary.histogram('det_output_16x_stat', det_y))
            tf.add_to_collection('activation_summary', tf.summary.histogram('cl_output_16x_stat', cl_y))

            self._model['output'] = self._model['det_output_16x'], self._model['cl_output_16x']
