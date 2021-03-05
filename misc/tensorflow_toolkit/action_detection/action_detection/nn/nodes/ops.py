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

import numpy as np
import tensorflow as tf

from action_detection.nn.nodes.variables import create_variable


def assign_moving_average(variable, value, momentum):
    """Creates Op to assign exponentially decayed variable value.

    :param variable: Input variable
    :param value: New value
    :param momentum: Momentum scalar value
    :return: Update op
    """

    with tf.name_scope(None, 'AssignMovingAvg'):
        return tf.assign_sub(variable, (variable - value) * (1.0 - momentum))


def assign_moving_frequency(variable, value, momentum):
    """Creates Op to assign normalized variable value

    :param variable: Input normalized variable
    :param value: New distribution value
    :param momentum: Momentum scalar value
    :return: Update op
    """

    with tf.name_scope(None, 'AssignMovingFreq'):
        averaged_value = float(momentum) * variable + value * (1.0 - momentum)
        sum_values = tf.reduce_sum(averaged_value)
        normalizer = tf.cond(tf.greater(sum_values, 0.0), lambda: tf.reciprocal(sum_values), lambda: 1.0)
        return tf.assign(variable, normalizer * averaged_value)


def batch_norm(input_value, name, is_training, momentum=0.99, num_channels=None,
               trg_weights=None, trg_weight_var=None, trg_bias=None,
               reduction_axis=None, add_summary=False):
    """Adds Batch Normalization (BN) after the specified input tensor.

    :param input_value: Input tensor
    :param name: Name of block
    :param is_training: Inference mode
    :param momentum: Momentum scalar value
    :param num_channels: Input channels number
    :param trg_weights: Dictionary with weight variables if needed
    :param trg_weight_var: Dictionary with weight variables if needed
    :param trg_bias: Dictionary with bias variables if needed
    :param reduction_axis: List of axis for reduction
    :param add_summary: Whether to add summary info
    :return: Normalized tensor
    """

    if num_channels is None:
        num_channels = [input_value.get_shape()[-1]]

    if reduction_axis is None:
        reduction_axis = [0, 1, 2]

    if trg_weight_var is not None and trg_weights is not None and trg_bias is not None:
        with tf.variable_scope(name):
            beta = tf.get_variable('beta', num_channels, tf.float32, tf.initializers.constant(0.0), trainable=True)
            gamma = tf.get_variable('gamma', num_channels, tf.float32, tf.initializers.constant(1.0), trainable=True)
            mean = tf.get_variable(
                'moving_mean', num_channels, tf.float32, tf.initializers.constant(0.0),
                synchronization=tf.VariableSynchronization.ON_READ, trainable=False,
                aggregation=tf.VariableAggregation.MEAN)
            variance = tf.get_variable(
                'moving_variance', num_channels, tf.float32, tf.initializers.constant(1.0),
                synchronization=tf.VariableSynchronization.ON_READ, trainable=False,
                aggregation=tf.VariableAggregation.MEAN)

            scales = tf.multiply(tf.rsqrt(variance), gamma)
            if trg_weights.get_shape()[-1] == num_channels:
                # weights shape is [h, w, in, out]
                new_weights = tf.multiply(trg_weights, tf.reshape(scales, [1, 1, 1, -1]))
            elif trg_weights.get_shape()[-1] == 1:
                # weights shape is [h, w, in, 1]
                new_weights = tf.multiply(trg_weights, tf.reshape(scales, [1, 1, -1, 1]))
            else:
                # weights shape is [h, w, out, in]
                trans_weights = tf.transpose(trg_weights, [0, 1, 3, 2])
                scaled_trans_weights = tf.multiply(trans_weights, tf.reshape(scales, [1, 1, 1, -1]))
                new_weights = tf.transpose(scaled_trans_weights, [0, 1, 3, 2])
            new_bias = beta - tf.multiply(scales, mean)

            weights_update_op = tf.assign(trg_weight_var, new_weights)
            bias_update_op = tf.assign(trg_bias, new_bias)

            tf.add_to_collection('bn_merge_ops', weights_update_op)
            tf.add_to_collection('bn_merge_ops', bias_update_op)

            return input_value
    else:
        with tf.variable_scope(name):
            beta = tf.get_variable('beta', num_channels, tf.float32, tf.initializers.constant(0.0), trainable=True)
            gamma = tf.get_variable('gamma', num_channels, tf.float32, tf.initializers.constant(1.0), trainable=True)

            if add_summary:
                tf.add_to_collection('bn_summary', tf.summary.histogram(beta.op.name, beta))
                tf.add_to_collection('bn_summary', tf.summary.histogram(gamma.op.name, gamma))

        def _get_train_values():
            with tf.variable_scope(name):
                batch_mean, batch_variance = tf.nn.moments(input_value, reduction_axis, name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', num_channels, tf.float32, tf.initializers.constant(0.0),
                    synchronization=tf.VariableSynchronization.ON_READ, trainable=False,
                    aggregation=tf.VariableAggregation.MEAN)
                moving_variance = tf.get_variable(
                    'moving_variance', num_channels, tf.float32, tf.initializers.constant(1.0),
                    synchronization=tf.VariableSynchronization.ON_READ, trainable=False,
                    aggregation=tf.VariableAggregation.MEAN)

                if add_summary:
                    tf.add_to_collection('bn_summary', tf.summary.histogram('moving_mean', moving_mean))
                    tf.add_to_collection('bn_summary', tf.summary.histogram('moving_variance', moving_variance))

                mean_update_op = assign_moving_average(moving_mean, batch_mean, momentum)
                var_update_op = assign_moving_average(moving_variance, batch_variance, momentum)

                with tf.control_dependencies([mean_update_op, var_update_op]):
                    return tf.identity(batch_mean), tf.identity(batch_variance)

        def _get_test_values(reuse=True):
            with tf.variable_scope(name, reuse=reuse):
                out_mean = tf.get_variable(
                    'moving_mean', num_channels, tf.float32, tf.initializers.constant(0.0),
                    synchronization=tf.VariableSynchronization.ON_READ, trainable=False,
                    aggregation=tf.VariableAggregation.MEAN)
                out_variance = tf.get_variable(
                    'moving_variance', num_channels, tf.float32, tf.initializers.constant(1.0),
                    synchronization=tf.VariableSynchronization.ON_READ, trainable=False,
                    aggregation=tf.VariableAggregation.MEAN)

            return out_mean, out_variance

        if is_training is not None:
            mean, variance = tf.cond(is_training, _get_train_values, _get_test_values)
        else:
            mean, variance = _get_test_values(False)

        output_value = tf.nn.batch_normalization(input_value, mean, variance, beta, gamma, 1e-3)
        output_value.set_shape(input_value.get_shape())

        return output_value


def kernel_norm(kernel, axis, merge_op=False, merge_op_transit=False):
    """Adds Op to kerry out L2 normalization over model weights instead of weight decay.

    :param kernel: Input weights
    :param axis: List of axis to merge
    :param merge_op: Whether to run with merged normalization
    :param merge_op_transit: Whether to run in merging mode
    :return: Normalized weights
    """

    if merge_op and not merge_op_transit:
        out_kernel = kernel
    else:
        out_kernel = tf.nn.l2_normalize(kernel, axis=axis)

    return out_kernel


def conv2d(input_value, shape, name, fn_activation=None, stride=None, rate=None, padding='SAME', depth_wise=False,
           use_bn=True, use_bias=False, use_dropout=None, dropout_shape=None, is_training=None, reuse_var=None,
           init=None, init_scale=1.0, add_summary=False, merge_op=False, merge_op_transit=False,
           norm_kernel=False, pr_product=False):
    """Wrapper for Convolution layer.

    :param input_value: Input tensor
    :param shape: Kernel shape
    :param name: Name of block
    :param fn_activation: Output activation function if needed
    :param stride: Stride size
    :param rate: Dilation rate
    :param padding: Padding size
    :param depth_wise: Whether to carry out depth-wise convolution
    :param use_bn: Whether to add Batch Normalization stage
    :param use_bias: Whether to add bias term
    :param use_dropout: Whether to use dropout regularization
    :param dropout_shape: Shape of Bernoulli variable
    :param is_training: Whether to run in training mode
    :param reuse_var: Whether to reuse variables
    :param init: Initialization type
    :param init_scale: Scale of init weights
    :param add_summary: Whether to add summary info
    :param merge_op: Whether to run with merged normalization ops
    :param merge_op_transit: Whether to run in merging mode
    :param norm_kernel: Whether to normalize weights
    :param pr_product: Whether to use PR-Product
    :return: Output tensor
    """

    if init is None:
        init = tf.initializers.variance_scaling(scale=init_scale, mode='fan_in', distribution='truncated_normal')

    with tf.variable_scope(name, reuse=reuse_var) as scope:
        if depth_wise:
            out_num_channels = shape[-2]
            kernel_var = create_variable('weights', shape, init)
            if norm_kernel:
                kernel = kernel_norm(kernel_var, [0, 1], merge_op, merge_op_transit)
            else:
                kernel = kernel_var
            conv = tf.nn.depthwise_conv2d(input_value, kernel, padding=padding,
                                          strides=stride if stride is not None else [1, 1, 1, 1],
                                          rate=rate if rate is not None else [1, 1])
        else:
            out_num_channels = shape[-1]
            kernel_var = create_variable('weights', shape, init)
            if norm_kernel:
                kernel = kernel_norm(kernel_var, [0, 1, 2], merge_op, merge_op_transit)
            else:
                kernel = kernel_var
            conv = tf.nn.conv2d(input_value, kernel, padding=padding,
                                strides=stride if stride is not None else [1, 1, 1, 1],
                                dilations=rate if rate is not None else [1, 1, 1, 1])

            if pr_product and is_training is not None:
                if shape[0] != 1 or shape[1] != 1:
                    raise Exception('PR Product support 1x1 conv only.')

                prod = conv
                with tf.name_scope('pr_product'):
                    kernel_norms = tf.ones([1, 1, 1, out_num_channels], dtype=tf.float32) \
                        if norm_kernel else tf.reduce_sum(tf.square(kernel), axis=[0, 1, 2], keepdims=True)
                    input_norms = tf.reduce_sum(tf.square(input_value), axis=-1, keepdims=True)

                    prod_norms = kernel_norms * input_norms
                    alpha = tf.sqrt(tf.square(prod_norms) - tf.square(prod))

                    conv = tf.stop_gradient(alpha / prod_norms) * prod + \
                           tf.stop_gradient(prod / prod_norms) * (prod_norms - alpha)

        if add_summary:
            weights_summary_op = tf.summary.histogram(kernel.op.name + '/values', kernel)
            tf.add_to_collection('weights_summary', weights_summary_op)

        if merge_op:
            biases = create_variable('biases', [out_num_channels], tf.initializers.constant(0.0))

            if merge_op_transit:
                conv = batch_norm(conv, 'bn', is_training, num_channels=out_num_channels,
                                  trg_weights=kernel, trg_weight_var=kernel_var, trg_bias=biases)
            conv = tf.nn.bias_add(conv, biases)
        else:
            if use_bias:
                biases = create_variable('biases', [out_num_channels], tf.initializers.constant(0.0))
                conv = tf.nn.bias_add(conv, biases)

                if add_summary:
                    biases_summary_op = tf.summary.histogram(biases.op.name, biases)
                    tf.add_to_collection('weights_summary', biases_summary_op)

            if use_bn:
                conv = batch_norm(conv, 'bn', is_training, num_channels=out_num_channels)

        if fn_activation is not None:
            conv = fn_activation(conv, name=scope.name+'_fn_activation')

        if use_dropout is not None and is_training is not None:
            conv = tf.cond(is_training,
                           lambda: tf.nn.dropout(conv, use_dropout, dropout_shape),
                           lambda: conv)

    return conv


def max_pool(input_value, kernel=None, stride=None, k=2, padding='SAME', name='max_pool'):
    """Wrapper for max-pooling operation.

    :param input_value: Input tensor
    :param kernel: Kernel shape
    :param stride: Stride sizes
    :param k: Stride factor
    :param padding: Padding sizes
    :param name: Name of block
    :return: Output tensor
    """

    assert k is not None or kernel is not None and stride is not None,\
        "The kernel and stride should be specified if k is None"

    if kernel is not None:
        ksize = kernel
    else:
        ksize = [1, k, k, 1]

    if stride is not None:
        strides = stride
    else:
        strides = [1, k, k, 1]

    with tf.name_scope(name):
        pool = tf.nn.max_pool(input_value, ksize=ksize, strides=strides, padding=padding)

    return pool


def glob_max_pool(input_value, name='glob_max_pool', add_summary=False):
    """Wrapper for global max-pooling operation.

    :param input_value: Input tensor
    :param name: Name of block
    :param add_summary: Whether to add summary info
    :return: Output tensor
    """

    with tf.name_scope(name):
        pool = tf.reduce_max(input_value, axis=[1, 2], keepdims=True)
        if add_summary:
            biases_summary_op = tf.summary.histogram(pool.op.name, pool)
            tf.add_to_collection('activation_summary', biases_summary_op)

        return pool


def dropout(input_value, keep_prob, noise_shape=None, is_training=None):
    """Wrapper for dropout regularization

    :param input_value: Input tensor
    :param keep_prob: Probability to preserve value
    :param noise_shape: Shape of dropout variable
    :param is_training: Whether to run regularization
    :return: Output tensor
    """

    if is_training is not None:
        return tf.cond(is_training,
                       lambda: tf.nn.dropout(input_value, keep_prob=keep_prob, noise_shape=noise_shape),
                       lambda: input_value)
    else:
        return input_value


def extract_gt_values(values, labels, name):
    """Selects values according specified labels

    :param values: Input tensor of [n, m] shape
    :param labels: Input tensor of [n] shape. Each value in [0, m - 1] range
    :param name: Name of block
    :return: Selected by ID values of [n] shape
    """

    with tf.name_scope(name, 'gt_values'):
        valid_logits = tf.batch_gather(values, tf.reshape(labels, [-1, 1]))
        flat_logits = tf.reshape(valid_logits, [-1])
    return flat_logits


def safe_masked_reduce_op(input_value, mask, reduce_op, default_value=0.0):
    """Carry out safe (if empty) reduction operation over masked values.

    :param input_value: Input tensor
    :param mask: Mask of valid values
    :param reduce_op: Reduction op
    :param default_value: Value to return if all values are invalid
    :return: Output scalar value
    """

    def _process():
        masked_x = tf.boolean_mask(input_value, mask)
        return reduce_op(masked_x)

    input_size = tf.size(input_value)
    num_masked = tf.reduce_sum(tf.cast(mask, tf.int32))
    is_valid = tf.logical_and(tf.greater(input_size, 0),
                              tf.greater(num_masked, 0))

    return tf.cond(is_valid,
                   lambda: _process(),
                   lambda: default_value)


def safe_reduce_op(input_value, reduce_op, default_value=0.0):
    """Carry out safe (if empty) reduction operation.

    :param input_value: Input tensor
    :param reduce_op: Reduction op
    :param default_value: Value to return if all values are invalid
    :return: Output scalar value
    """

    return tf.cond(tf.greater(tf.size(input_value), 0),
                   lambda: reduce_op(input_value),
                   lambda: default_value)


def interpolate_extreme_points(embeddings, num_samples, name, top_fraction=0.3):
    """Extracts further from the geometric center points and samples interpolated new values.

    :param embeddings: Input embedding vectors
    :param num_samples: Number of samples to generate
    :param name: Name of block
    :param top_fraction: Number of further embeddings to interpolate between
    :return: Interpolated embeddings
    """

    with tf.name_scope(name):
        def _process():
            center = tf.reduce_mean(embeddings, axis=0)
            norm_center = tf.reshape(tf.nn.l2_normalize(center), [-1, 1])

            dist_to_center = tf.reshape(1.0 - tf.matmul(embeddings, norm_center), [-1])

            num_corner_points = tf.maximum(tf.cast(top_fraction * tf.cast(num_embeddings, tf.float32), tf.int32), 2)
            _, corner_points_ids = tf.nn.top_k(dist_to_center, num_corner_points, sorted=False)

            left_ids = tf.random.uniform([num_samples], minval=0, maxval=num_corner_points, dtype=tf.int32)
            right_ids = tf.random.uniform([num_samples], minval=0, maxval=num_corner_points-1, dtype=tf.int32)
            right_ids = tf.where(tf.less(right_ids, left_ids), right_ids, right_ids + 1)

            glob_left_ids = tf.gather(corner_points_ids, left_ids)
            glob_right_ids = tf.gather(corner_points_ids, right_ids)

            left_embeddings = tf.gather(embeddings, glob_left_ids)
            right_embeddings = tf.gather(embeddings, glob_right_ids)

            cos_gamma = tf.reduce_sum(left_embeddings * right_embeddings, axis=1)
            sin_gamma = tf.sqrt(1.0 - tf.square(cos_gamma))

            ratio = tf.random.uniform([num_samples], minval=0.0, maxval=1.0, dtype=tf.float32)
            alpha_angle = tf.acos((1.0 - ratio) / tf.sqrt(ratio * ratio - 2.0 * ratio * cos_gamma + 1.0)) + \
                          tf.atan(ratio * sin_gamma / (ratio * cos_gamma - 1.0))
            gamma_angle = tf.acos(cos_gamma)
            betta_angle = gamma_angle - alpha_angle

            left_scale = tf.stop_gradient(tf.reshape(tf.sin(betta_angle) / sin_gamma, [-1, 1]))
            right_scale = tf.stop_gradient(tf.reshape(tf.sin(alpha_angle) / sin_gamma, [-1, 1]))

            return left_scale * left_embeddings + right_scale * right_embeddings

        num_embeddings = tf.shape(embeddings)[0]
        is_valid = tf.greater(num_embeddings, 1)

        return tf.cond(is_valid,
                       lambda: _process(),
                       lambda: tf.zeros([0, embeddings.get_shape()[1]], dtype=tf.float32))


def sample_from_extreme_points(input_embeddings, input_labels, num_classes, num_samples):
    """Carry out sampling of new embeddings for each class by interpolating existing border points.

    :param input_embeddings: Input embeddings
    :param input_labels: Class labels of embeddings
    :param num_classes: Number of classes
    :param num_samples: Number of samples per class
    :return: Samples embeddings
    """

    out_embeddings = []
    out_labels = []

    for class_id in xrange(num_classes):
        class_mask = tf.equal(input_labels, class_id)
        class_embeddings = tf.boolean_mask(input_embeddings, class_mask)

        sampled_embeddings = interpolate_extreme_points(class_embeddings, num_samples,
                                                        'sampling/class_{}'.format(class_id))

        out_embeddings.append(sampled_embeddings)
        out_labels.append(tf.fill([tf.shape(sampled_embeddings)[0]], class_id))

    return tf.concat(out_embeddings, axis=0), tf.concat(out_labels, axis=0)


def gsoftmax(logits, axis=-1, weight=1.0):
    """Calculates Generalized Softmax over the input logits

    :param logits: Input logits
    :param axis: Axis to reduce
    :param weight: Mixing weight
    :return: Distribution tensor
    """

    assert axis == -1

    num_classes = logits.get_shape()[-1]
    factor = -0.5 * np.sqrt(2.0)

    with tf.variable_scope(None, 'gsoftmax'):
        mean = tf.get_variable(
            'params/mean', [num_classes], tf.float32, tf.initializers.constant(0.0), trainable=True)
        log_sigma = tf.get_variable(
            'params/log_sigma', [num_classes], tf.float32, tf.initializers.constant(0.0), trainable=True)

        gaussian_cdf = 0.5 * (tf.erf(factor * (mean - logits) * tf.exp(tf.negative(log_sigma))) + 1.0)

        return tf.nn.softmax(logits + weight * gaussian_cdf)
