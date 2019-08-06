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

from action_detection.nn.backbones.base_backbone import BaseBackbone
from action_detection.nn.nodes.ops import batch_norm, conv2d, max_pool, dropout


class RMNet(BaseBackbone):
    """Base class for RMNet architecture.
    """

    def __init__(self, net_input, fn_activation, is_training, merge_bn, merge_bn_transition, name='rmnet',
                 use_extra_layers=False, keep_prob=0.9, reduced=False, norm_kernels=False):
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

        super(RMNet, self).__init__(net_input, fn_activation, is_training, merge_bn, merge_bn_transition, name,
                                    use_extra_layers, keep_prob, reduced, norm_kernels)

    def _normalize_inputs(self, input_value, input_depth):
        """Carry out normalization of network input.

        :param input_value: Network input
        :param input_depth: Number of input channels
        :return: Normalized network input
        """

        with tf.variable_scope('init_norm'):
            out = batch_norm(input_value, 'bn', self._is_training, num_channels=input_depth)
        return out

    def _init_block(self, input_value, num_channels, name):
        """Converts network input in some internal representation.

        :param input_value: Network input
        :param num_channels: Number of input and output channels
        :param name: Name of node
        :return: Tensor
        """

        with tf.variable_scope(name):
            out = conv2d(input_value, [3, 3, num_channels[0], num_channels[1]], 'dim_inc_conv',
                         stride=[1, 2, 2, 1], is_training=self._is_training, init=self._ort_init,
                         use_bias=False, use_bn=True, fn_activation=self._fn_activation,
                         merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                         add_summary=True, init_scale=0.1, norm_kernel=self._norm_kernels)

        return out

    def _bottleneck(self, input_value, num_channels, name, keep_prob=None, rate=None, factor=4., k=1):
        """

        :param input_value:
        :param num_channels: Number of input and output channels
        :param name: Name of node
        :param keep_prob: Probability to keep value in dropout
        :param rate: Rate value for dilated convolutions
        :param factor: Factor to reduce number of channels in bottleneck
        :param k: Stride of node
        :return: Tensor
        """

        with tf.variable_scope(name):
            internal_num_channels = int(num_channels[1] // factor)

            conv1 = conv2d(input_value, [1, 1, num_channels[0], internal_num_channels], 'dim_red',
                           stride=[1, 1, 1, 1], is_training=self._is_training, init=self._ort_init,
                           use_bias=False, use_bn=True, fn_activation=self._fn_activation,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                           norm_kernel=self._norm_kernels)

            conv2 = conv2d(conv1, [3, 3, internal_num_channels, 1], 'inner_conv', init_scale=0.1,
                           stride=[1, k, k, 1], depth_wise=True, rate=rate, is_training=self._is_training,
                           use_bias=False, use_bn=True, fn_activation=self._fn_activation,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                           norm_kernel=self._norm_kernels)

            conv3 = conv2d(conv2, [1, 1, internal_num_channels, num_channels[1]], 'dim_inc', init_scale=0.1,
                           is_training=self._is_training, use_bias=False, use_bn=True, fn_activation=None,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                           norm_kernel=self._norm_kernels)

            if keep_prob is not None:
                conv3 = dropout(conv3, keep_prob, is_training=self._is_training)

            skip_branch = input_value if k == 1 else max_pool(input_value, k=k)
            if num_channels[0] != num_channels[1]:
                skip_branch = conv2d(skip_branch, [1, 1, num_channels[0], num_channels[1]], 'skip_conv',
                                     stride=[1, 1, 1, 1], is_training=self._is_training, init_scale=0.1,
                                     use_bias=False, use_bn=True, fn_activation=None,
                                     merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                                     norm_kernel=self._norm_kernels)

            out = tf.add(conv3, skip_branch)

            out = self._fn_activation(out)

        return out

    def _build(self, input_value, use_extra_layers, name, keep_prob):
        """Constructs target network.

        :param input_value: Network input
        :param use_extra_layers: Whether to enable extra layers
        :param name: Name of output network
        :param keep_prob: Probability to keep value in dropout
        """

        with tf.variable_scope(name):
            tf.add_to_collection('activation_summary', tf.summary.histogram('data_stat', input_value))

            local_y = self._normalize_inputs(input_value, 3)
            tf.add_to_collection('activation_summary', tf.summary.histogram('input_stat', local_y))

            # Init block: x1 -> x1/2
            local_y = self._init_block(local_y, [3, 32], 'init_block')
            self._model['output_init'] = local_y
            tf.add_to_collection('activation_summary', tf.summary.histogram('init_block_stat', local_y))

            # Bottleneck1: x1/2 -> x1/2
            local_y = self._bottleneck(local_y, [32, 32], 'bottleneck1_1', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [32, 32], 'bottleneck1_2', keep_prob=keep_prob)
            if not self._reduced:
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
            if not self._reduced:
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_7', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [64, 64], 'bottleneck2_8', keep_prob=keep_prob)
            self._model['output_4x'] = local_y
            tf.add_to_collection('activation_summary', tf.summary.histogram('output_4x_stat', local_y))

            # Bottleneck3: x1/4 -> x1/8
            local_y = self._bottleneck(local_y, [64, 128], 'bottleneck3_0', keep_prob=keep_prob, k=2)
            local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_1', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_2', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_3', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_4', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_5', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_6', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_7', keep_prob=keep_prob)
            if not self._reduced:
                local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_8', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_9', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [128, 128], 'bottleneck3_10', keep_prob=keep_prob)
            self._model['output_8x'] = local_y
            tf.add_to_collection('activation_summary', tf.summary.histogram('output_8x_stat', local_y))

            # Bottleneck4: x1/8 -> x1/16
            local_y = self._bottleneck(local_y, [128, 256], 'bottleneck4_0', keep_prob=keep_prob, k=2)
            local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_1', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_2', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_3', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_4', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_5', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_6', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_7', keep_prob=keep_prob)
            local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_8', keep_prob=keep_prob)
            if not self._reduced:
                local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_9', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_10', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [256, 256], 'bottleneck4_11', keep_prob=keep_prob)
            self._model['output_16x'] = local_y
            tf.add_to_collection('activation_summary', tf.summary.histogram('output_16x_stat', local_y))

            if use_extra_layers:
                # Bottleneck5: x1/16 -> x1/32
                local_y = self._bottleneck(local_y, [256, 512], 'bottleneck5_0', keep_prob=keep_prob, k=2)
                local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_1', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_2', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_3', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_4', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_5', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_6', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_7', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_8', keep_prob=keep_prob)
                local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_9', keep_prob=keep_prob)
                if not self._reduced:
                    local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_10', keep_prob=keep_prob)
                    local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_11', keep_prob=keep_prob)
                    local_y = self._bottleneck(local_y, [512, 512], 'bottleneck5_12', keep_prob=keep_prob)
                self._model['output_32x'] = local_y
                tf.add_to_collection('activation_summary', tf.summary.histogram('output_32x_stat', local_y))

            self._model['output'] = local_y
