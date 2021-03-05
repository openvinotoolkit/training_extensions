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

from __future__ import print_function

from abc import ABCMeta

import tensorflow as tf


class BaseNetwork(object):
    """Base class for task-specific networks.
    """

    __metaclass__ = ABCMeta

    def __init__(self, is_training, merge_bn, merge_bn_transition, lr_params, wd_weight, global_step,
                 use_nesterov, norm_kernels):
        """Constructor.

        :param is_training: Training indicator variable
        :param merge_bn: Whether to run with merged BatchNorms
        :param merge_bn_transition: Whether to run in BatchNorm merging mode
        :param lr_params: Learning rate parameters
        :param wd_weight: Weight decay value
        :param global_step: Variable for counting the training steps if exists
        :param use_nesterov: Whether to enable nesterov momentum calculation
        :param norm_kernels: Whether to normalize convolution kernels
        """

        self._is_training = is_training
        self._merge_bn = merge_bn
        self._merge_bn_transition = merge_bn_transition
        self._weight_decay = wd_weight
        self._lr_params = lr_params
        self._use_nesterov = use_nesterov
        self._norm_kernels = norm_kernels

        self._max_grad_norm = 1.0
        self._momentum = 0.9
        self._global_step = tf.Variable(0, trainable=False, name='GlobalStep') if global_step is None else global_step

        self._model = {}

    @staticmethod
    def _add_loss_summary(loss, name):
        """Adds summary of specified scalar loss into collection of summary.

        :param loss: Scalar loss value
        :param name: Name of loss in collection
        """

        tf.add_to_collection('loss_summary', tf.summary.scalar(name, loss))

    def _create_lr_schedule(self):
        """Creates learning rate scheduled variable.

        :return: Learning rate variable
        """

        with tf.name_scope('learning_rate'):
            if 'schedule' not in self._lr_params or self._lr_params['schedule'] == 'piecewise_constant':
                learning_rate = tf.train.piecewise_constant(
                    self._global_step, self._lr_params['boundaries'], self._lr_params['values'], name='lr_value')
            elif self._lr_params['schedule'] == 'cosine_decay_restarts':
                learning_rate = tf.train.cosine_decay_restarts(
                    self._lr_params['init_value'], self._global_step, self._lr_params['first_decay_steps'],
                    self._lr_params['t_mul'], self._lr_params['m_mul'], self._lr_params['alpha'],
                    name='lr_value')
            else:
                raise Exception('Unknown lt schedule: {}'.format(self._lr_params['schedule']))

            tf.add_to_collection('lr_summary', tf.summary.scalar('lr', learning_rate))
            self._model['lr'] = learning_rate

    def create_optimizer(self):
        """Create default optimizer.

        :return: Optimizer
        """

        return tf.train.MomentumOptimizer(learning_rate=self._model['lr'],
                                          momentum=self._momentum,
                                          use_nesterov=self._use_nesterov)

    @staticmethod
    def load_available_variables(checkpoint_path, sess, checkpoint_scope, model_scope,
                                 print_vars=False, extra_ends=None):
        """Loads all matched by name variables from the specified checkpoint.

        :param checkpoint_path: Path to checkpoint
        :param sess: Session to load in
        :param checkpoint_scope: Name of scope to load from
        :param model_scope: Name of scope to load in
        :param print_vars: Whether to print loaded names of variables
        :param extra_ends: List of name suffix to match too
        """

        def _print_variables(var_list, header):
            """Prints variable names

            :param var_list: List of variables or names
            :param header: Header to print
            """

            if print_vars and len(var_list) > 0:
                if isinstance(var_list[0], basestring):
                    var_names = var_list
                else:
                    var_names = [var_name.name[:-2] for var_name in var_list]

                sorted_var_names = var_names
                sorted_var_names.sort()

                print('\n{}:'.format(header))
                for var_name in sorted_var_names:
                    print('   {}'.format(var_name))

        reader = tf.train.NewCheckpointReader(checkpoint_path)
        checkpoint_var_names = reader.get_variable_to_shape_map().keys()

        valid_ends = ['weights', 'biases', 'beta', 'gamma', 'moving_mean', 'moving_variance']
        if extra_ends is not None:
            valid_ends += extra_ends
        valid_ends = tuple(valid_ends)

        valid_checkpoint_var_names = [n for n in checkpoint_var_names
                                      if n.startswith(checkpoint_scope) and n.endswith(valid_ends)]
        _print_variables(valid_checkpoint_var_names, 'Checkpoint variables')

        all_model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)
        valid_model_vars = [v for v in all_model_vars if v.name[:-2].endswith(valid_ends)]
        _print_variables(valid_model_vars, 'Session variables')

        valid_model_vars_map = {}
        for model_var in valid_model_vars:
            out_name = model_var.name[:-2].replace(model_scope, checkpoint_scope)
            if out_name in valid_checkpoint_var_names:
                valid_model_vars_map[out_name] = model_var
            elif print_vars:
                print('Unmatched variable: {}'.format(out_name))
        _print_variables(list(valid_model_vars_map), 'Matched variables')

        if len(valid_model_vars_map) == 0:
            raise Exception('The provided checkpoint or source model scope does not contain valid variables')

        loader = tf.train.Saver(var_list=valid_model_vars_map)
        loader.restore(sess, checkpoint_path)
        print('{} / {} (from checkpoint\'s {}) model parameters have been restored.'
              .format(len(valid_model_vars_map), len(valid_model_vars), len(valid_checkpoint_var_names)))

    @staticmethod
    def get_merge_train_op():
        """Creates Op to merge all internal update ops into single node.

        :return: Merged Op
        """

        bn_merge_ops = tf.get_collection('bn_merge_ops')

        if len(bn_merge_ops) > 0:
            out_op = tf.group(bn_merge_ops)
        else:
            out_op = tf.no_op()

        return out_op

    @property
    def total_loss(self):
        """Returns total loss Op.

        :return: Total loss op
        """

        return self._model['total_loss']
