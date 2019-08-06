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
from action_detection.nn.nodes.ops import safe_masked_reduce_op, sample_from_extreme_points, extract_gt_values,\
    gsoftmax, assign_moving_frequency


def weight_decay(variables, weight, name, key_word='weights'):
    """Creates Op to carry out weight decay.

    :param variables: List of variables
    :param weight: Weights of loss
    :param name: Name of loss
    :param key_word: Name to filter variables
    :return: Loss value
    """

    with tf.name_scope(name):
        vars_to_decay = [var for var in variables if key_word in var.name]
        if len(vars_to_decay) > 0:
            weight_decays = [tf.reduce_mean(tf.square(var)) for var in vars_to_decay]
            normalizer = weight / float(len(vars_to_decay))
            out_loss = tf.multiply(tf.add_n(weight_decays), normalizer, name='weight_decay')
            return out_loss
        else:
            return 0.0


def unit_gamma(variables, weight, name, key_word='gamma'):
    """Creates Op to preserve gamma parameter for Batch Norm to unit value.

    :param variables: List of variables
    :param weight: Weights of loss
    :param name: Name of loss
    :param key_word: Name to filter variables
    :return: Loss value
    """

    with tf.name_scope(name):
        vars_to_decay = [var for var in variables if key_word in var.name]
        if len(vars_to_decay) > 0:
            weight_decays = [tf.reduce_mean(tf.square(1.0 - var)) for var in vars_to_decay]
            normalizer = weight / float(len(vars_to_decay))
            out_loss = tf.multiply(tf.add_n(weight_decays), normalizer, name='unit_gamma')
            return out_loss
        else:
            return 0.0


def orthogonal_conv(variables, weight, name, key_word='weights', key_scope=''):
    """Creates Op to fit kernel parameters onto orthogonal matrix.

    :param variables: List of variables
    :param weight: Weight of loss
    :param name: Name of loss
    :param key_word: Name to filter variables
    :param key_scope: Name of scope to filter variables
    :return: Loss value
    """

    with tf.name_scope(name):
        conv_vars = [var for var in variables if key_word in var.name]
        if key_scope is not None and key_scope != '':
            conv_vars = [var for var in conv_vars if key_scope in var.name]

        ort_losses = []
        for var in conv_vars:
            var_shape = var.get_shape()
            if len(var_shape) != 4:
                continue

            kernel_h = int(var_shape[0])
            kernel_w = int(var_shape[1])

            input_size = kernel_h * kernel_w * int(var_shape[2])
            output_size = int(var_shape[3])

            model_weights = tf.reshape(var, [input_size, output_size])
            coefficients = tf.matmul(tf.transpose(model_weights, perm=[1, 0]), model_weights)

            ort_loss = tf.reduce_sum(tf.square(tf.matrix_set_diag(
                coefficients, tf.zeros([output_size], dtype=tf.float32))))
            ort_losses.append(ort_loss)

        if len(ort_losses) > 0:
            normalizer = weight / float(len(ort_losses))
            out_loss = tf.multiply(tf.add_n(ort_losses), normalizer, name='orthogonal_conv')
            return out_loss
        else:
            return 0.0


def decorrelate_features(input_value, weight, name):
    """Create Op to orthogonalize input features

    :param input_value: Input features
    :param weight: Weight of loss
    :param name: Name of block
    :return: Loss value
    """

    with tf.name_scope(name):
        tensor_shape = tf.shape(input_value)

        batch_size = tensor_shape[0]
        input_size = tensor_shape[1] * tensor_shape[2]
        output_size = tensor_shape[3]

        batched_data = tf.reshape(input_value, [-1, input_size, output_size])
        coefficients = tf.matmul(tf.transpose(batched_data, [0, 2, 1]), batched_data)

        loss_value = tf.reduce_mean(tf.square(tf.matrix_set_diag(
            coefficients, tf.zeros([batch_size, output_size], dtype=tf.float32))))
        out_loss = tf.multiply(loss_value, weight, name='decorrelate_features')

    return out_loss


def adaptive_weighting(values, name, scale=2.0, add_summary=True, exclude_invalid=True, margin=0.1, init_value=2.5):
    """Carry out adaptive weighting (with learnable parameter) for the input values.

    :param values: Input values
    :param name: Name of block
    :param scale: Scale of target values
    :param add_summary: Whether to add summary info
    :param exclude_invalid: Whether to exclude invalid or negative values from output
    :param margin: Margin for the output
    :param init_value: Init value of parameters
    :return: Adaptive weighted sum of input values
    """

    num_variables = len(values)

    with tf.variable_scope(name):
        init_params = np.full([num_variables], init_value, dtype=np.float32)
        params = create_variable('params/weights', init_params.shape, init_params)

        clipped_values = tf.maximum(tf.stack(values), 0.0)
        weights = tf.exp(tf.negative(params))
        weighted_values = weights * clipped_values

        out_values = params + scale * weighted_values

        if exclude_invalid:
            out_value = safe_masked_reduce_op(out_values, tf.greater(values, 0.0), tf.reduce_sum)
        else:
            out_value = tf.reduce_sum(out_values, 0.0)
        out_value = tf.maximum(margin + out_value, 0.0)

        if add_summary:
            for var_id in xrange(num_variables):
                tf.add_to_collection('loss_summary', tf.summary.scalar('values/var_{}'
                                                                       .format(var_id), clipped_values[var_id]))
                tf.add_to_collection('loss_summary', tf.summary.scalar('weights/var_{}'
                                                                       .format(var_id), weights[var_id]))

    return out_value


def frequencies_weighting(counts, values, decay, name, add_summary=True, limits=None):
    """Carry out weighting of input values according their frequencies.

    :param counts: Counts of af each value
    :param values: Input values
    :param decay: Decay value
    :param name: Name of block
    :param add_summary: Whether to add summary info
    :param limits: Limits to restrict weights in format: [min, max]
    :return: Weighted sum of input values
    """

    assert len(counts) == len(values)

    num_values = len(counts)
    assert num_values > 0

    if limits is not None:
        assert len(limits) == 2
        assert 0.0 <= limits[0] < limits[1]

    with tf.variable_scope(name):
        values = tf.stack(values)
        counts = tf.stack(counts)

        sum_counts = tf.reduce_sum(counts)
        normalizer = tf.cond(tf.greater(sum_counts, 0.0), lambda: tf.reciprocal(sum_counts), lambda: 1.0)
        frequencies = normalizer * counts

        smoothed_frequencies = tf.get_variable('smoothed_frequencies', [num_values], tf.float32,
                                               tf.initializers.constant(1.0 / num_values), trainable=False)
        update_op = assign_moving_frequency(smoothed_frequencies, frequencies, 1.0 - decay)
        with tf.control_dependencies([update_op]):
            out_frequencies = tf.identity(smoothed_frequencies)

        norm_factor = tf.reduce_sum(out_frequencies) / float(num_values)
        weights = tf.where(tf.greater(out_frequencies, 0.0),
                           norm_factor / out_frequencies,
                           tf.zeros_like(out_frequencies))

        if limits is not None:
            weights = tf.maximum(limits[0], tf.minimum(weights, limits[1]))

        weighted_values = normalizer * tf.stop_gradient(weights) * values
        out_loss = tf.reduce_sum(weighted_values)

        if add_summary:
            for class_id in xrange(num_values):
                tf.add_to_collection('loss_summary', tf.summary.scalar('weight/var_{}'.format(class_id),
                                                                       weights[class_id]))
                tf.add_to_collection('loss_summary', tf.summary.scalar('freq/var_{}'.format(class_id),
                                                                       smoothed_frequencies[class_id]))
                tf.add_to_collection('loss_summary', tf.summary.scalar('in_val/var_{}'.format(class_id),
                                                                       values[class_id]))
                tf.add_to_collection('loss_summary', tf.summary.scalar('out_val/var_{}'.format(class_id),
                                                                       weighted_values[class_id]))

    return out_loss


def class_frequencies_weighting(losses, labels, mask, num_classes, name, add_summary=True, decay=0.9, limits=None):
    """Wrapper for class weighting loss according frequencies of classes.

    :param losses: List of all losses
    :param labels: List of labels
    :param mask: Mask of valid values
    :param num_classes: Number of classes
    :param name: Name of block
    :param add_summary: Whether to add summary info
    :param decay: Decay scalar value
    :param limits: Limits to restrict weights in format: [min, max]
    :return: Loss value
    """

    with tf.variable_scope(name):
        class_sum_losses = []
        class_counts = []
        for class_id in xrange(num_classes):
            class_mask = tf.logical_and(tf.equal(labels, class_id), mask)

            class_sum_losses.append(safe_masked_reduce_op(losses, class_mask, tf.reduce_sum))
            class_counts.append(tf.reduce_sum(tf.cast(class_mask, tf.float32)))

        out_loss = frequencies_weighting(class_counts, class_sum_losses, decay, 'balancing',
                                         add_summary=add_summary, limits=limits)

    return out_loss


def max_entropy_ce_loss(labels, logits, entropy_weight, name, add_summary=True, eps=1e-12,
                        enable_gsoftmax=False, gsoftmax_weight=1.0):
    """Calculates Cross-Entropy loss which is regularized by Max-Entropy term.

    :param labels: List of labels
    :param logits: List of class logits
    :param entropy_weight: Weight of Entropy term
    :param name: Name of block
    :param add_summary: Whether to add summary info
    :param eps: Epsilon scalar value
    :param enable_gsoftmax: Whether to enable General Softmax calculation
    :param gsoftmax_weight: Weight parameter for the General Softmax
    :return: List of loss values
    """

    with tf.name_scope(name):
        if enable_gsoftmax:
            probs = gsoftmax(logits, axis=-1, weight=gsoftmax_weight)
            ce_losses = tf.negative(tf.log(extract_gt_values(probs, labels, 'gt_probs')))
        else:
            probs = tf.nn.softmax(logits, axis=-1)
            ce_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='ce_losses')

        neg_entropy = tf.reduce_sum(probs * tf.log(probs + eps), axis=-1)
        out_losses = tf.maximum(ce_losses + entropy_weight * neg_entropy, 0.0)

        if add_summary:
            sparsity = tf.nn.zero_fraction(out_losses)
            tf.add_to_collection('loss_summary', tf.summary.scalar('ce_losses/sparsity', sparsity))

    return out_losses


def focal_ce_loss(labels, logits, alpha, gamma, name, enable_gsoftmax=False, gsoftmax_weight=1.0):
    """Calculates Focal+Cross-Entropy loss. See: https://arxiv.org/abs/1708.02002

    :param labels: List of labels
    :param logits: List of class logits
    :param alpha: Loss scale
    :param gamma: Power parameter
    :param name: Name of block
    :param enable_gsoftmax: Whether to enable General Softmax calculation
    :param gsoftmax_weight: Weight parameter for the General Softmax
    :return: List of loss values
    """

    with tf.name_scope(name):
        if enable_gsoftmax:
            probs = gsoftmax(logits, axis=-1, weight=gsoftmax_weight)
        else:
            probs = tf.nn.softmax(logits, axis=-1)

        gt_probs = extract_gt_values(probs, labels, 'gt_probs')
        out_losses = float(-alpha) * tf.pow(1.0 - gt_probs, float(gamma)) * tf.log(gt_probs)

    return out_losses


def gradient_harmonized_ce_loss(labels, logits, name, num_bins=20, momentum=0.75, smooth_param=0.1,
                                add_summary=True, enable_gsoftmax=False, gsoftmax_weight=1.0):
    """Calculates Cross-Entropy loss which is normalized by inversing gradient histogram.
       See: https://arxiv.org/abs/1811.05181

    :param labels: List of labels
    :param logits: List of class logits
    :param name: Name of block
    :param num_bins: Number of histogram bins
    :param momentum: Momentum scalar value
    :param smooth_param: Parameter to smooth distribution
    :param add_summary: Whether to add summary info
    :param enable_gsoftmax: Whether to enable General Softmax calculation
    :param gsoftmax_weight: Weight parameter for the General Softmax
    :return: List of loss values
    """

    assert num_bins > 1
    assert smooth_param >= 0.0

    all_edges = np.array([float(x) / float(num_bins) for x in range(num_bins + 1)], dtype=np.float32)
    all_edges[-1] += 1e-6
    range_starts = tf.constant(all_edges[:-1].reshape([1, -1]), dtype=tf.float32)
    range_ends = tf.constant(all_edges[1:].reshape([1, -1]), dtype=tf.float32)

    with tf.variable_scope(name):
        if enable_gsoftmax:
            probs = gsoftmax(logits, axis=-1, weight=gsoftmax_weight)
        else:
            probs = tf.nn.softmax(logits, axis=-1)

        gt_probs = extract_gt_values(probs, labels, 'gt_probs')

        errors = tf.stop_gradient(tf.reshape(1.0 - gt_probs, [-1, 1]))
        mask = tf.logical_and(tf.greater_equal(errors, range_starts), tf.less(errors, range_ends))
        float_mask = tf.cast(mask, tf.float32)
        range_ids = tf.argmax(float_mask, axis=1)

        bins_sizes = tf.reduce_sum(float_mask, axis=0)
        total_num = tf.reduce_sum(bins_sizes)
        frequencies = bins_sizes / tf.maximum(1.0, total_num)

        smoothed_frequencies = tf.get_variable('smoothed_frequencies', [num_bins], tf.float32,
                                               tf.initializers.constant(1.0 / num_bins), trainable=False)
        update_op = assign_moving_frequency(smoothed_frequencies, frequencies, momentum)
        with tf.control_dependencies([update_op]):
            out_frequencies = tf.identity(smoothed_frequencies)

        bin_weights = tf.where(tf.greater(out_frequencies, 0.0),
                               1.0 / (out_frequencies + smooth_param),
                               tf.zeros_like(out_frequencies))
        out_weights = tf.gather(bin_weights, range_ids)

        out_losses = out_weights * tf.negative(tf.log(gt_probs))

        if add_summary:
            tf.add_to_collection('loss_summary', tf.summary.histogram('gh_errors', errors))

    return out_losses


def adaptive_scale(scale, labels, logits, num_classes, name=None, add_summary=True):
    """Adaptive scale Op to use with Cross-Entropy loss. See: https://arxiv.org/abs/1905.00292

    :param scale: Input scale variable to update
    :param labels: List of labels
    :param logits:  List of class logits
    :param num_classes: Number of classes
    :param name: Name of block
    :param add_summary: Whether to add summary info
    :return: Scale scalar value
    """

    with tf.name_scope(name, 'adaptive_scale'):
        gt_cos_values = extract_gt_values(logits, labels, 'gt_cos_values')
        gt_angles = tf.acos(gt_cos_values)
        median_gt_angle = tf.reduce_mean(gt_angles)

        exp_values = tf.exp(scale * logits)

        valid_mask = tf.one_hot(labels, num_classes, True, False, dtype=tf.bool)
        invalid_exp_values = tf.where(valid_mask, tf.zeros_like(exp_values), exp_values)

        mean_exp_value = tf.reduce_sum(invalid_exp_values) / tf.cast(tf.maximum(1, tf.size(labels)), tf.float32)

        new_scale = tf.stop_gradient(tf.log(mean_exp_value) / tf.cos(tf.minimum(np.pi / 4.0, median_gt_angle)))
        update_op = tf.assign(scale, new_scale)
        with tf.control_dependencies([update_op]):
            out_scale = tf.identity(scale)

        if add_summary:
            tf.add_to_collection('loss_summary', tf.summary.scalar('adaptive_scale/angle', median_gt_angle))
            tf.add_to_collection('loss_summary', tf.summary.scalar('adaptive_scale/mean_exp', mean_exp_value))

    return out_scale


def weighted_ce_loss(labels, logits, num_classes, name, max_entropy_weight=0.4, add_summary=True,
                     decay=0.9, limits=None, alpha=None, gamma=None, num_bins=None):
    """Wrapper to calculate Cross-Entropy loss with some regularization term.

    :param labels: List of labels
    :param logits: List of class logits
    :param num_classes: Number of classes
    :param name: Name of block
    :param max_entropy_weight: Weight of Entropy term
    :param add_summary: Whether to add summary info
    :param decay: Decay scalar value
    :param limits: Limits to restrict weights in format: [min, max]
    :param alpha: Focal loss scale parameter
    :param gamma: Focal loss power parameter
    :param num_bins: GH loss number of bins
    :return: Weighted sum of CE losses
    """

    enable_max_entropy_loss = max_entropy_weight is not None and max_entropy_weight > 0.0
    enable_focal_loss = alpha is not None and alpha > 0.0 and gamma is not None and gamma > 0.0
    enable_gradientharmonized_loss = num_bins is not None and num_bins > 0
    if enable_max_entropy_loss and enable_focal_loss and enable_gradientharmonized_loss:
        raise Exception('Cannot enable different CE losses simultaneously')

    with tf.variable_scope(name):
        if enable_max_entropy_loss:
            ce_losses = max_entropy_ce_loss(labels, logits, max_entropy_weight, 'me_ce_losses', add_summary)
        elif enable_focal_loss:
            ce_losses = focal_ce_loss(labels, logits, alpha, gamma, 'fl_ce_losses')
        elif enable_gradientharmonized_loss:
            ce_losses = gradient_harmonized_ce_loss(labels, logits, 'gh_ce_losses', num_bins, add_summary=add_summary)
        else:
            ce_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='ce_losses')

        valid_ce_losses_mask = tf.greater(ce_losses, 0.0)

        out_loss = class_frequencies_weighting(ce_losses, labels, valid_ce_losses_mask, num_classes, 'by_class',
                                               add_summary=add_summary, decay=decay, limits=limits)

    return out_loss


def class_balanced_hard_losses(all_losses, class_counts, top_k, name=None, add_summary=True):
    """Carry out selection of hardest losses for each class.

    :param all_losses: List of all losses by class
    :param class_counts: Number of valid losses for each class
    :param top_k: Max number of losses to select
    :param name: Name of block
    :param add_summary: Whether to add summary info
    :return: Loss value
    """

    assert 0.0 < top_k < 1.0

    with tf.name_scope(name, 'balanced_hard_losses'):
        class_counts = tf.stack(tuple(class_counts), axis=0)
        valid_mask = tf.greater(class_counts, 0.0)
        mean_valid_count = tf.maximum(1, tf.cast(safe_masked_reduce_op(class_counts, valid_mask, tf.reduce_mean),
                                                 tf.int32))

        top_losses = []
        for class_losses in all_losses:
            top_losses.append(
                tf.cond(tf.greater(tf.size(class_losses), mean_valid_count),
                        lambda: tf.nn.top_k(class_losses, mean_valid_count)[0],  # pylint: disable=cell-var-from-loop
                        lambda: class_losses))  # pylint: disable=cell-var-from-loop

        top_losses = tf.concat(tuple(top_losses), axis=0)
        num_valid_classes = tf.reduce_sum(tf.cast(valid_mask, tf.float32))

        out_losses = tf.cond(
            tf.greater(num_valid_classes, 0),
            lambda: tf.nn.top_k(
                top_losses,
                tf.maximum(1, tf.cast(float(top_k) * tf.cast(tf.size(top_losses), tf.float32), tf.int32)))[0],
            lambda: tf.zeros([0], tf.float32))

        out_loss = safe_masked_reduce_op(out_losses, tf.greater(out_losses, 0.0), tf.reduce_mean)

        if add_summary:
            tf.add_to_collection('loss_summary', tf.summary.scalar('hard_losses/mean_num',
                                                                   mean_valid_count))
            tf.add_to_collection('loss_summary', tf.summary.scalar('hard_losses/sparsity',
                                                                   tf.nn.zero_fraction(top_losses)))

    return out_loss


def explicit_center_loss(embeddings, labels, centers, name, top_k=0.5, min_class_size=2, add_summary=True):
    """Calculates center loss for the specified embedding vectors.

    :param embeddings: Embeddings vectors
    :param labels: List of labels
    :param centers: Centers of classes
    :param name: Name of block
    :param top_k: Max number of losses per class
    :param min_class_size: Min number of samples per class
    :param add_summary: Whether to add summary info
    :return: Loss value
    """

    def _process(mask, center):
        class_embeddings = tf.boolean_mask(embeddings, mask)
        losses = 1.0 - tf.matmul(class_embeddings, tf.reshape(center, [-1, 1]))
        return tf.reshape(losses, [-1])

    with tf.variable_scope(name):
        class_losses = []
        class_counts = []
        for class_id in xrange(len(centers)):
            class_mask = tf.equal(labels, class_id)
            class_size = tf.reduce_sum(tf.cast(class_mask, tf.int32))

            is_valid_class = tf.greater_equal(class_size, min_class_size)

            class_losses.append(tf.cond(is_valid_class,
                                        lambda: _process(class_mask, centers[class_id]),  # pylint: disable=cell-var-from-loop
                                        lambda: tf.zeros([0], tf.float32)))
            class_counts.append(tf.cond(is_valid_class,
                                        lambda: tf.cast(class_size, tf.float32),  # pylint: disable=cell-var-from-loop
                                        lambda: 0.0))

        out_loss = class_balanced_hard_losses(class_losses, class_counts, top_k, add_summary=add_summary)

    return out_loss


def explicit_pull_push_loss(embeddings, labels, centers, margin, name, top_k=0.5, add_summary=True):
    """Calculates Pull-Push loss with Smart margin. See: https://arxiv.org/abs/1812.02465

    :param embeddings: Embedding vectors
    :param labels: List of labels
    :param centers: Centers of classes
    :param margin: Margin scalar value
    :param name: Name of block
    :param top_k: Max number of losses per class
    :param add_summary: Whether to add summary info
    :return: Loss value
    """

    def _calculate_loss(embd, class_name):
        valid_center = tf.reshape(centers[class_name], [-1, 1])
        pos_dist = tf.reshape(1.0 - tf.matmul(embd, valid_center), [-1])

        invalid_centers = tf.concat(tuple([tf.reshape(centers[i], [-1, 1]) for i in xrange(len(centers))
                                           if i != class_name]), axis=1)
        neg_dist = tf.reduce_min(1.0 - tf.matmul(embd, invalid_centers), axis=1)

        return tf.maximum(0.0, margin + pos_dist - neg_dist)

    with tf.variable_scope(name):
        out_losses = []
        class_counts = []
        for class_id in xrange(len(centers)):
            class_mask = tf.equal(labels, class_id)
            class_embeddings = tf.boolean_mask(embeddings, class_mask)

            class_losses = tf.cond(tf.greater_equal(tf.reduce_sum(tf.cast(class_mask, tf.int32)), 1),
                                   lambda: _calculate_loss(class_embeddings, class_id),  # pylint: disable=cell-var-from-loop
                                   lambda: tf.zeros([0], tf.float32))
            out_losses.append(class_losses)

            class_valid_losses_mask = tf.greater(class_losses, 0.0)
            class_num_valid = tf.reduce_sum(tf.cast(class_valid_losses_mask, tf.int32))
            class_counts.append(tf.cast(class_num_valid, tf.float32))

        out_loss = class_balanced_hard_losses(out_losses, class_counts, top_k, add_summary=add_summary)

    return out_loss


def sampling_losses(input_embeddings, input_labels, centers, num_samples, main_weight, auxiliary_weight, name,
                    logit_scale=None, max_entropy_weight=0.4, add_summary=True, limits=None,
                    alpha=None, gamma=None, num_bins=None):
    """Calculates Cross_entropy, Center and Push losses for the sampled embeddings.

    :param input_embeddings: Embedding vectors
    :param input_labels: List labels
    :param centers: Centers of classes
    :param num_samples: Number of instances per class to sample
    :param main_weight: CE loss weight
    :param auxiliary_weight: Auxiliary loss weight
    :param name: Name of block
    :param logit_scale: Scale for CE loss
    :param max_entropy_weight: Weight of Entropy term for CE loss
    :param add_summary: Whether to add summary info
    :param limits: Weight limits
    :param alpha: Focal loss scale parameter
    :param gamma: Focal loss power parameter
    :param num_bins: Number of bins for GH loss
    :return: Loss value
    """

    def _process(embd, labels):
        sampled_logits = tf.matmul(embd, tf.stack(centers, axis=1))
        scaled_logits = logit_scale * sampled_logits if logit_scale is not None else sampled_logits
        ce_loss_value = weighted_ce_loss(labels, scaled_logits, len(centers), 'ce_loss', max_entropy_weight,
                                         limits=limits, alpha=alpha, gamma=gamma, num_bins=num_bins)

        push_loss_value = local_push_loss(embd, labels, len(centers), 0.5, 'local_push_loss')
        center_loss_value = explicit_center_loss(embd, labels, centers, 'center_loss', add_summary=add_summary)
        auxiliary_loss = adaptive_weighting([center_loss_value, push_loss_value], 'auxiliary', add_summary=add_summary)

        return main_weight * ce_loss_value + auxiliary_weight * auxiliary_loss

    with tf.variable_scope(name):
        sampled_embeddings, sampled_labels = \
            sample_from_extreme_points(input_embeddings, input_labels, len(centers), num_samples)

        num_embeddings = tf.shape(sampled_embeddings)[0]
        is_valid = tf.greater(num_embeddings, 0)

        return tf.cond(is_valid,
                       lambda: _process(sampled_embeddings, sampled_labels),
                       lambda: 0.0)


def local_push_loss(embeddings, labels, num_classes, margin, name, top_k=0.5):
    """Calculates Push losses for each pair of classes and carry out two-stage hard sample mining over them.

    :param embeddings: Embedding vectors
    :param labels: List of labels
    :param num_classes: Number of classes
    :param margin: Margin scalar value
    :param name: Name of block
    :param top_k: Max number of losses
    :return: Loss value
    """

    def _process(mask_a, mask_b):
        embeddings_a = tf.boolean_mask(embeddings, mask_a)
        embeddings_b = tf.boolean_mask(embeddings, mask_b)

        distance_matrix = 1.0 - tf.matmul(embeddings_a, tf.transpose(embeddings_b))
        losses = tf.reshape(margin - distance_matrix, [-1])

        num_valid_values = tf.reduce_sum(tf.cast(tf.greater(losses, 0.0), tf.float32))

        return losses, num_valid_values

    class_pairs = [(i, j) for i in xrange(num_classes) for j in range(i + 1, num_classes)]

    with tf.variable_scope(name):
        all_losses = []
        class_counts = []
        for class_i, class_j in class_pairs:
            class_i_mask = tf.equal(labels, class_i)
            class_j_mask = tf.equal(labels, class_j)

            is_valid = tf.logical_and(tf.reduce_any(class_i_mask), tf.reduce_any(class_j_mask))
            class_losses, out_count = tf.cond(is_valid,
                                              lambda: _process(class_i_mask, class_j_mask),  # pylint: disable=cell-var-from-loop
                                              lambda: (tf.zeros([0], tf.float32), 0.0))

            all_losses.append(class_losses)
            class_counts.append(out_count)

        out_loss = class_balanced_hard_losses(all_losses, class_counts, top_k)

    return out_loss


def multi_similarity_loss(embeddings, labels, alpha=2.0, beta=50.0, gamma=1.0, similarity_delta=0.1, name=None):
    """Calculates Multi-similarity loss. See: https://arxiv.org/abs/1904.06627

    :param embeddings: Embedding vectors
    :param labels: List of labels
    :param alpha: Method parameter
    :param beta: Method parameter
    :param gamma: Shift parameter
    :param similarity_delta: Delta parameter
    :param name: Name of block
    :return: Loss value
    """

    with tf.variable_scope(name, 'multi_similarity_loss'):
        similarity_matrix = tf.matmul(embeddings, tf.transpose(embeddings))
        shifted_similarity_matrix = similarity_matrix - float(gamma)

        same_label_mask = tf.equal(tf.reshape(labels, [-1, 1]), tf.reshape(labels, [1, -1]))
        different_label_mask = tf.logical_not(same_label_mask)

        pos_sim_matrix = tf.where(same_label_mask, similarity_matrix, tf.ones_like(similarity_matrix))
        neg_sim_matrix = tf.where(different_label_mask, similarity_matrix, -1.0 * tf.ones_like(similarity_matrix))

        hardest_pos_threshold = tf.reduce_min(pos_sim_matrix, axis=-1) - similarity_delta
        hardest_neg_threshold = tf.reduce_max(neg_sim_matrix, axis=-1) + similarity_delta

        positives_mask = tf.logical_and(tf.less(similarity_matrix, hardest_neg_threshold), same_label_mask)
        negatives_mask = tf.logical_and(tf.greater(similarity_matrix, hardest_pos_threshold), different_label_mask)

        positive_components = tf.where(positives_mask,
                                       tf.exp(float(-alpha) * shifted_similarity_matrix),
                                       tf.zeros_like(shifted_similarity_matrix))
        negative_components = tf.where(negatives_mask,
                                       tf.exp(float(beta) * shifted_similarity_matrix),
                                       tf.zeros_like(shifted_similarity_matrix))

        positive_values = float(1.0 / alpha) * tf.log(1.0 + tf.reduce_sum(positive_components, axis=-1))
        negative_values = float(1.0 / beta) * tf.log(1.0 + tf.reduce_sum(negative_components, axis=-1))

        out_loss = tf.reduce_mean(positive_values + negative_values)

    return out_loss


def balanced_l1_loss(input_value, alpha, gamma):
    """Calculates Balanced L1 loss. See: https://arxiv.org/abs/1904.02701

    :param input_value: Input values
    :param alpha: Method parameter
    :param gamma: Method parameter
    :return: Re-weighted loss values
    """

    betta = np.exp(float(gamma) / float(alpha)) - 1.0
    shift = float(gamma) / float(betta) - float(alpha)

    with tf.name_scope('balanced_l1_loss'):
        abs_x = tf.abs(input_value)
        out_loss = tf.where(tf.less(abs_x, 1.0),
                            alpha / betta * (betta * abs_x + 1.0) * tf.log(betta * abs_x + 1.0) - alpha * abs_x,
                            gamma * abs_x + shift)

    return out_loss
