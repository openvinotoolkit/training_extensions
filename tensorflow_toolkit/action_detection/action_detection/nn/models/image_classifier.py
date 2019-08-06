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

from action_detection.nn.models.base_network import BaseNetwork
from action_detection.nn.backbones import get_backbone, get_orthogonal_scope_name
from action_detection.nn.nodes.ops import conv2d, glob_max_pool
from action_detection.nn.nodes.losses import weight_decay, unit_gamma, orthogonal_conv, decorrelate_features


class ImageClassifier(BaseNetwork):
    """Describes network for the image classification problem.
    """

    def __init__(self, backbone_name, net_input, labels, fn_activation, is_training, num_classes, merge_bn,
                 merge_bn_transition, lr_params, wd_weight=1e-2, global_step=None, keep_probe=None,
                 name='image_classifier', use_nesterov=True, norm_kernels=False):
        """Constructor.

        :param backbone_name: Name of target backbone
        :param net_input: Input images
        :param labels: Labels of input images
        :param fn_activation: Main activation function of network
        :param is_training: Training indicator variable
        :param num_classes: Number of image classes
        :param merge_bn: Whether to run with merged BatchNorms
        :param merge_bn_transition: Whether to run in BatchNorm merging mode
        :param lr_params: Learning rate parameters
        :param wd_weight: Weight decay value
        :param global_step: Variable for counting the training steps if exists
        :param keep_probe: Probability to keep values in dropout
        :param name: Network name
        :param use_nesterov: Whether to enable nesterov momentum calculation
        :param norm_kernels: Whether to normalize convolution kernels
        """

        super(ImageClassifier, self).__init__(is_training, merge_bn, merge_bn_transition, lr_params, wd_weight,
                                              global_step, use_nesterov, norm_kernels)

        self._fn_activation = fn_activation
        self._num_classes = num_classes
        self._name = name
        self._keep_probe = keep_probe

        self._model['input'] = net_input
        self._model['labels'] = labels

        self._build_network(net_input, backbone_name)
        if is_training is not None:
            self._create_lr_schedule()
            self._build_losses(labels, backbone_name)

    def _classifier_layers(self, input_value, num_inputs):
        """Create nodes to carry out classification into target number of classes.

        :param input_value: Input features
        :param num_inputs: Number of input channels
        :param keep_prob: Probability to keep values in dropout
        :return: Class logits
        """

        with tf.variable_scope('classifier_layer'):
            conv1 = conv2d(input_value, [1, 1, num_inputs, 1024], 'dim_inc', fn_activation=self._fn_activation,
                           use_bias=False, use_bn=True, is_training=self._is_training,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                           add_summary=True, norm_kernel=self._norm_kernels)

            glob_pool = glob_max_pool(conv1, add_summary=True)

            conv2 = conv2d(glob_pool, [1, 1, 1024, 1280], 'internal', fn_activation=self._fn_activation,
                           use_bias=False, use_bn=True, is_training=self._is_training,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                           add_summary=True, norm_kernel=self._norm_kernels)
            logits = conv2d(conv2, [1, 1, 1280, self._num_classes], 'logits', fn_activation=None,
                            use_bias=True, use_bn=False, is_training=self._is_training,
                            merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                            add_summary=True, norm_kernel=self._norm_kernels)
            logits = tf.reshape(logits, [-1, self._num_classes], name='output')

            return logits

    def _build_network(self, input_value, backbone_name):
        """Creates parameterized network architecture.

        :param input_value: Input tensor
        :param backbone_name: Target name of backbone
        """

        with tf.variable_scope(self._name):
            self._feature_extractor = get_backbone(backbone_name, input_value, self._fn_activation, self._is_training,
                                                   self._merge_bn, self._merge_bn_transition, use_extra_layers=True,
                                                   name=backbone_name, keep_probe=self._keep_probe,
                                                   norm_kernels=self._norm_kernels)
            extracted_features = self._feature_extractor.output

            logits = self._classifier_layers(extracted_features, num_inputs=extracted_features.get_shape()[3])
            self._model['output'] = logits
            tf.add_to_collection('activation_summary', tf.summary.histogram('logits', logits))

    def _build_losses(self, labels, backbone_name):
        """Adds losses to the training graph.

        :param labels: Labels of the image
        :param backbone_name: Target backbone name
        """

        assert 'output' in self._model.keys(), 'Call _build_network method before.'

        with tf.name_scope(self._name + '_losses'):
            all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)

            ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=self._model['output'], name='ce_losses'), name='ce_loss')
            self._add_loss_summary(ce_loss, 'ce_loss')

            wd_loss = weight_decay(all_trainable_vars, self._weight_decay, 'var_reg')
            self._add_loss_summary(wd_loss, 'wd_loss')

            unit_gamma_loss = unit_gamma(all_trainable_vars, 0.1 * self._weight_decay, 'gamma_reg')
            self._add_loss_summary(unit_gamma_loss, 'gamma_loss')

            orthogonal_loss = orthogonal_conv(all_trainable_vars, 0.25 * self._model['lr'], 'ort_reg',
                                              key_scope=get_orthogonal_scope_name(backbone_name))
            self._add_loss_summary(orthogonal_loss, 'orthogonal_loss')

            skeleton = self._feature_extractor.skeleton
            feature_names = ['output_init', 'output_2x', 'output_4x', 'output_8x', 'output_16x', 'output_32x']
            feature_maps = [skeleton[n] for n in feature_names]
            feature_map_weights = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
            decorr_losses = [decorrelate_features(f, feature_map_weights[i] * self._model['lr'],
                                                  'decor_{}'.format(feature_names[i]))
                             for i, f in enumerate(feature_maps)]
            for i, feature_name in enumerate(feature_names):
                self._add_loss_summary(decorr_losses[i], 'decor_{}'.format(feature_name))

            decorr_loss = tf.add_n(decorr_losses, name='fm_reg')
            self._add_loss_summary(decorr_loss, 'decorr_loss')

            total_loss = tf.add_n([ce_loss, wd_loss, unit_gamma_loss, orthogonal_loss, decorr_loss],
                                  name='total_loss')
            self._add_loss_summary(total_loss, 'total_loss')
            self._model['total_loss'] = total_loss

    def num_valid_op(self, labels):
        """Returns Op to calculate number of valid predictions.

        :param labels: Target image labels
        :return: Number of valid predictions Op
        """

        with tf.name_scope(self._name + '_accuracy_op'):
            is_correct = tf.to_float(tf.equal(tf.argmax(self._model['output'], axis=-1),
                                              tf.cast(labels, tf.int64)))
            return tf.reduce_sum(is_correct)

    def predictions(self):
        """Returns model predictions.

        :return: Model predictions
        """

        with tf.name_scope(self._name + '_prediction_op'):
            return tf.argmax(self._model['output'], axis=-1)
