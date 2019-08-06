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

from action_detection.nn.models import SSD
from action_detection.nn.backbones import get_backbone
from action_detection.nn.nodes.losses import adaptive_weighting, weighted_ce_loss, sampling_losses, \
    explicit_center_loss, explicit_pull_push_loss, local_push_loss
from action_detection.nn.nodes.ops import conv2d

class ActionNet(SSD):
    """Describes network for the person detection (PD) and action recognition (AR) problems.
    """

    def __init__(self, backbone_name, net_input, labels, annot, fn_activation, is_training,
                 head_params, merge_bn, merge_bn_transition, lr_params, mbox_param, action_params,
                 wd_weight=1e-2, global_step=None, name='actionnet', use_nesterov=True, norm_kernels=False):
        """Constructor.

        :param backbone_name: Name of target backbone
        :param net_input: Input images
        :param labels: Bbox classes
        :param annot: Bbox coordinates
        :param fn_activation: Main activation function of network
        :param is_training: Training indicator variable
        :param head_params: Parameters of SSD heads
        :param merge_bn: Whether to run with merged BatchNorms
        :param merge_bn_transition: Whether to run in BatchNorm merging mode
        :param lr_params: Learning rate parameters
        :param mbox_param: Parameters for SSD-based PD training
        :param action_params: Parameters for AR training
        :param wd_weight: Weight decay value
        :param global_step: Variable for counting the training steps if exists
        :param name: Network name
        :param use_nesterov: Whether to enable nesterov momentum calculation
        :param norm_kernels: Whether to normalize convolution kernels
        """

        assert backbone_name == 'twinnet'

        self._action_params = action_params

        super(ActionNet, self).__init__(backbone_name, net_input, labels, annot, fn_activation, is_training,
                                        head_params, merge_bn, merge_bn_transition, lr_params, mbox_param, wd_weight,
                                        global_step, name, use_nesterov, norm_kernels)

    def _add_head_shared(self, input_value, out_size, name):
        """Adds shared part of the action head.

        :param input_value: Input tensor
        :param out_size: Output number of channels
        :param name: Name of block
        :return: Output tensor
        """

        with tf.variable_scope(name):
            conv1 = conv2d(input_value, [1, 1, input_value.get_shape()[3], out_size], 'conv1',
                           fn_activation=self._fn_activation, norm_kernel=self._norm_kernels,
                           use_bias=False, use_bn=True, is_training=self._is_training,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition)
            conv2 = conv2d(conv1, [3, 3, out_size, 1], 'conv2', norm_kernel=self._norm_kernels,
                           depth_wise=True, use_bias=False, use_bn=True, is_training=self._is_training,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition)

        return conv2

    def _add_anchor_specific_head(self, input_value, input_size, out_size, name):
        """Adds anchor-specific part of the action head.

        :param input_value: Input tensor
        :param input_size: Input number of channels
        :param out_size: Output number of channels
        :param name: Name of block
        :return: Output tensor
        """

        with tf.variable_scope(name):
            conv1 = conv2d(input_value, [1, 1, input_size, out_size], 'conv1',
                           use_bias=False, use_bn=True, is_training=self._is_training,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                           add_summary=True, norm_kernel=self._norm_kernels)
            action_branch = tf.nn.l2_normalize(conv1, axis=-1)

        return action_branch

    def _add_action_classifier_body(self, input_value, input_size, out_size, reuse=False):
        """Adds action classifier operations.

        :param input_value: Input tensor
        :param input_size: Input number of channels
        :param out_size: Embedding size
        :param reuse: Whether to reuse variables
        :return: Action logits
        """

        with tf.variable_scope('action_classifier'):
            conv1 = conv2d(input_value, [1, 1, input_size, input_size], 'conv1',
                           use_bias=False, use_bn=True, is_training=self._is_training,
                           fn_activation=self._fn_activation,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                           norm_kernel=self._norm_kernels, reuse_var=reuse)
            conv2 = conv2d(conv1, [1, 1, input_size, out_size], 'conv2',
                           use_bias=False, use_bn=True, is_training=self._is_training,
                           merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                           norm_kernel=self._norm_kernels, reuse_var=reuse)

            action_embeddings = tf.nn.l2_normalize(conv2, axis=-1)

        return action_embeddings

    def _add_action_classifier(self, input_value, input_size, out_size, pr_product, add_summary=True):
        """Adds embedding-based action classifier.

        :param input_value: Input tensor
        :param input_size: Input number of channels
        :param out_size: Embedding size
        :param pr_product: Whether to use PR-Product
        :param add_summary: Whether to store summary info
        :return: Action logits
        """

        action_embeddings = self._add_action_classifier_body(input_value, input_size, out_size)

        with tf.variable_scope('action_classifier'):
            self._model['action_embeddings'] = tf.reshape(action_embeddings, [-1, out_size])

            assert self._action_params['num_centers_per_action'] == 1
            action_centers = tf.stack(self._action_moving_centers, axis=1)
            self._model['action_centers'] = action_centers

            action_centers_kernel = tf.reshape(action_centers, [1, 1, out_size, self._action_params['num_actions']])
            all_action_logits = tf.nn.conv2d(action_embeddings, action_centers_kernel, [1, 1, 1, 1], 'SAME')

            if self._is_training is None and self._action_params['num_centers_per_action'] == 1:
                action_logits = tf.reshape(all_action_logits, [input_value.get_shape()[0], -1,
                                                               self._action_params['num_actions']],
                                           name='out_action_logits')
            else:
                if pr_product and self._is_training is not None:
                    prod = all_action_logits
                    alpha = tf.sqrt(1.0 - tf.square(prod))
                    all_action_logits = tf.stop_gradient(alpha) * prod + tf.stop_gradient(prod) * (1.0 - alpha)

                all_action_logits = tf.reshape(all_action_logits, [input_value.get_shape()[0], -1,
                                                                   self._action_params['num_actions'],
                                                                   self._action_params['num_centers_per_action']])
                self._model['all_action_logits'] = all_action_logits

                action_logits = tf.reduce_max(all_action_logits, axis=-1, name='out_action_logits')

                if add_summary:
                    action_center_ids = tf.reshape(tf.argmax(all_action_logits, axis=-1), [-1])
                    tf.add_to_collection('accuracy_summary',
                                         tf.summary.histogram('action_center_ids', action_center_ids))

            return action_logits

    def _build_train_action_heads(self, head_params, skeleton, action_prefix='cl_output_'):
        """Creates all action heads according twin detection branch.

        :param head_params: Detection head parameters
        :param skeleton: Network skeleton
        :param action_prefix: Name prefix for the new created heads
        """

        with tf.variable_scope('action_heads'):
            action_head_inputs = []
            for head_id, head_param in enumerate(head_params):
                cl_place_name = '{}{}x'.format(action_prefix, head_param.scale)
                action_x = skeleton[cl_place_name]

                head_y = self._add_head_shared(action_x, head_param.internal_size, 'head_{}_shared'.format(head_id + 1))

                anchor_heads = []
                for anchor_id in xrange(len(head_param.anchors)):
                    anchor_head = self._add_anchor_specific_head(head_y, head_param.internal_size,
                                                                 self._action_params['embedding_size'],
                                                                 'head_{}_anchor_{}'.format(head_id + 1, anchor_id + 1))
                    anchor_head = tf.expand_dims(anchor_head, axis=3)
                    anchor_heads.append(anchor_head)

                anchor_heads = tf.concat(anchor_heads, axis=3)
                anchor_heads = tf.reshape(anchor_heads,
                                          [anchor_heads.get_shape()[0], -1, self._action_params['embedding_size']])

                action_head_inputs.append(anchor_heads)

            action_head_inputs = tf.concat(action_head_inputs, axis=1)
            action_head_inputs = tf.expand_dims(action_head_inputs, axis=1)

            action_logits = self._add_action_classifier(action_head_inputs,
                                                        self._action_params['embedding_size'],
                                                        self._action_params['embedding_size'],
                                                        pr_product=True)

        self._model['action_logits'] = tf.identity(action_logits, name='out_action_logits')
        self._model['pr_action_conf'] = tf.nn.softmax(self._action_params['scale_end'] * action_logits, axis=-1,
                                                      name='out_action_probs')

    def _build_deploy_action_heads(self, head_params, skeleton, action_prefix='cl_output_'):
        """Creates all action heads according twin detection branch for deploying.

        :param head_params: Detection head parameters
        :param skeleton: Network skeleton
        :param action_prefix: Name prefix for the new created heads
        """

        assert self._action_params['num_centers_per_action'] == 1

        with tf.variable_scope('action_heads'):
            action_centers = tf.stack(self._action_moving_centers, axis=1)
            action_centers_kernel = tf.reshape(
                action_centers, [1, 1, self._action_params['embedding_size'], self._action_params['num_actions']])

            reuse_classifier_var = False
            all_head_logits = []
            for head_id, head_param in enumerate(head_params):
                cl_place_name = '{}{}x'.format(action_prefix, head_param.scale)
                action_x = skeleton[cl_place_name]

                shared_head_y = self._add_head_shared(
                    action_x, head_param.internal_size, 'head_{}_shared'.format(head_id + 1))

                all_anchor_logits = []
                for anchor_id in xrange(len(head_param.anchors)):
                    anchor_place_name = 'head_{}_anchor_{}'.format(head_id + 1, anchor_id + 1)

                    anchor_head = self._add_anchor_specific_head(shared_head_y, head_param.internal_size,
                                                                 self._action_params['embedding_size'],
                                                                 anchor_place_name)

                    anchor_embeddings = self._add_action_classifier_body(anchor_head,
                                                                         self._action_params['embedding_size'],
                                                                         self._action_params['embedding_size'],
                                                                         reuse_classifier_var)
                    reuse_classifier_var = True

                    anchor_logits = tf.nn.conv2d(anchor_embeddings, action_centers_kernel, [1, 1, 1, 1], 'SAME',
                                                 name='out_{}'.format(anchor_place_name))
                    all_anchor_logits.append(anchor_logits)

                all_head_logits.append(all_anchor_logits)

        self._model['deploy_action_logits'] = all_head_logits

    def _build_action_moving_centers(self):
        """Creates variables (not trainable by SGD) to store class centers.
        """

        with tf.variable_scope('action_moving_centers'):
            moving_centers = []
            for center_class_id in xrange(self._action_params['num_actions']):
                init_center_value = tf.nn.l2_normalize(tf.random.normal(
                    [self._action_params['embedding_size']]), axis=-1)
                moving_centers.append(tf.get_variable(
                    'moving_center_{}'.format(center_class_id), initializer=init_center_value,
                    synchronization=tf.VariableSynchronization.ON_READ, trainable=False,
                    aggregation=tf.VariableAggregation.MEAN))

            self._action_moving_centers = moving_centers

    def _build_network(self, input_value, backbone_name):
        """Creates parameterized network architecture.

        :param input_value: Input tensor
        :param backbone_name: Target name of backbone
        """

        with tf.variable_scope(self._name):
            self._feature_extractor = get_backbone(backbone_name, input_value, self._fn_activation,
                                                   self._is_training, self._merge_bn, self._merge_bn_transition,
                                                   use_extra_layers=False, name=backbone_name,
                                                   norm_kernels=self._norm_kernels)
            skeleton = self._feature_extractor.skeleton

            self._build_action_moving_centers()
            self._build_detection_heads(self._head_params, skeleton, prefix='det_output_')

            if self._is_training is None and self._merge_bn:
                self._build_deploy_action_heads(self._head_params, skeleton)
            else:
                self._build_train_action_heads(self._head_params, skeleton)

    def _get_action_scale(self, start, end, num_steps, power):
        """Creates scheduled scale to train AR model part.

        :param start: Initial scale value
        :param end: Target scale value
        :param num_steps: Num steps for scale annealing
        :param power: Power parameter
        :return: Scale scalar value
        """

        if 'schedule' not in self._lr_params or self._lr_params['schedule'] == 'piecewise_constant':
            float_step = tf.cast(self._global_step, tf.float32)

            factor = float(end - start) / float(1 - power)
            var_a = factor / (float(num_steps) ** float(power))
            var_b = -factor * float(power) / float(num_steps)

            out_value = tf.cond(tf.less(self._global_step, num_steps),
                                lambda: var_a * tf.pow(float_step, float(power)) + var_b * float_step + float(start),
                                lambda: float(end))
        elif self._lr_params['schedule'] == 'cosine_decay_restarts':
            factor = start / self._lr_params['init_value']
            out_value = tf.maximum(factor * self._model['lr'], end)
        else:
            raise Exception('Unknown lt schedule: {}'.format(self._lr_params['schedule']))

        return out_value

    def _filter_matches_by_instance(self, gt_ids, matched_scores, matches_logits, matches_embeddings, labels):
        """Selects top-k matches per instance to train AR model part.

        :param gt_ids: Ground truth IDs
        :param matched_scores: Matched scores
        :param matches_logits: Matched action logits
        :param matches_embeddings: Matched action embedding vectors
        :param labels: Action labels
        :return: Filtered values
        """

        with tf.name_scope('instance_filtering_matches'):
            best_matches_mask, _ = self._get_top_matched_mask(
                gt_ids, matched_scores, self._action_params['matches_threshold'],
                self._action_params['max_num_samples_per_gt'], 'action_top',
                drop_ratio=self._action_params['sample_matches_drop_ratio'], soft_num_samples=False)

            matches_logits = tf.boolean_mask(matches_logits, best_matches_mask)
            matches_embeddings = tf.boolean_mask(matches_embeddings, best_matches_mask)
            matched_labels = tf.boolean_mask(labels, best_matches_mask)

        return matches_logits, matches_embeddings, matched_labels

    def _filter_matches_simple(self, matched_scores, matches_logits, matches_embeddings, labels):
        """Filter matches by score.

        :param matched_scores: Matched scores
        :param matches_logits: Matched action logits
        :param matches_embeddings: Matched action embedding vectors
        :param labels: Action labels
        :return: Filtered values
        """

        with tf.name_scope('simple_filtering_matches'):
            valid_matches_mask = tf.greater(matched_scores, self._action_params['matches_threshold'])
            self._add_loss_summary(tf.reduce_sum(tf.cast(valid_matches_mask, tf.int32)), 'num_matches')

            matches_logits = tf.boolean_mask(matches_logits, valid_matches_mask)
            matches_embeddings = tf.boolean_mask(matches_embeddings, valid_matches_mask)
            matched_labels = tf.boolean_mask(labels, valid_matches_mask)

        return matches_logits, matches_embeddings, matched_labels

    def _update_moving_centers(self, embeddings, labels, momentum=0.99, add_summary=True):
        """Creates Op to update class centers according batch center of each class.

        :param embeddings: Batch embeddings
        :param labels: Action labels
        :param momentum: Momentum scalar value
        :param add_summary: Whether to add summary info
        """

        def _constant_top_triangle_mask(num_values):
            mask = np.zeros([num_values, num_values], dtype=np.bool)
            mask[np.triu_indices(num_values, 1)] = True
            return mask

        def _estimate_new_center(embd, old_center):
            weights = 0.5 * (1.0 + tf.matmul(embd, tf.reshape(old_center, [-1, 1])))
            current_center = tf.nn.l2_normalize(tf.reduce_mean(weights * embd, axis=0))
            return tf.nn.l2_normalize(momentum * old_center + (1.0 - momentum) * current_center)

        with tf.variable_scope('update_moving_centers'):
            for class_id in xrange(self._action_params['num_actions']):
                class_mask = tf.equal(labels, class_id)
                class_embeddings = tf.boolean_mask(embeddings, class_mask)

                moving_class_center = self._action_moving_centers[class_id]
                new_center = tf.cond(tf.greater_equal(tf.reduce_sum(tf.cast(class_mask, tf.int32)), 1),
                                     lambda: _estimate_new_center(class_embeddings, moving_class_center),  # pylint: disable=cell-var-from-loop
                                     lambda: moving_class_center)  # pylint: disable=cell-var-from-loop

                center_update_op = tf.assign(moving_class_center, new_center)
                with tf.control_dependencies([center_update_op]):
                    out_class_center = tf.identity(moving_class_center)

                self._action_moving_centers[class_id] = out_class_center

            if add_summary:
                centers = tf.stack(self._action_moving_centers, axis=0)
                center_distances = 1.0 - tf.matmul(centers, tf.transpose(centers))
                top_triangle_distances = tf.boolean_mask(
                    center_distances, _constant_top_triangle_mask(self._action_params['num_actions']))

                self._add_loss_summary(tf.reduce_min(top_triangle_distances), 'centers/min_dist')
                self._add_loss_summary(tf.reduce_mean(top_triangle_distances), 'centers/mean_dist')
                self._add_loss_summary(tf.reduce_max(top_triangle_distances), 'centers/max_dist')

                for k in xrange(self._action_params['num_actions']):
                    for action_id in xrange(k + 1, self._action_params['num_actions']):
                        # noinspection PyUnresolvedReferences
                        self._add_loss_summary(center_distances[k, action_id],
                                               'centers/pair_{}_{}'.format(k, action_id))

    def _action_losses(self, matches_mask, matched_scores, gt_ids, labels):
        """Creates action-specific losses.

        :param matches_mask: Mask of matched priors
        :param matched_scores: IoU score of matches
        :param gt_ids: Ground truth IDs
        :param labels: Action labels
        :return: List of loss values
        """

        with tf.name_scope('action_loss'):
            matches_logits = tf.boolean_mask(self._model['action_logits'], matches_mask)
            matches_embeddings = tf.boolean_mask(self._model['action_embeddings'], tf.reshape(matches_mask, [-1]))

            # filter matched samples
            if self._action_params['max_num_samples_per_gt'] > 0:
                matches_logits, matches_embeddings, matched_labels = tf.cond(
                    tf.greater(tf.size(gt_ids), 0),
                    lambda: self._filter_matches_by_instance(
                        gt_ids, matched_scores, matches_logits, matches_embeddings, labels),
                    lambda: (matches_logits, matches_embeddings, labels))
            else:
                matches_logits, matches_embeddings, matched_labels = tf.cond(
                    tf.greater(tf.size(gt_ids), 0),
                    lambda: self._filter_matches_simple(
                        matched_scores, matches_logits, matches_embeddings, labels),
                    lambda: (matches_logits, matches_embeddings, labels))

            valid_action_mask = tf.not_equal(matched_labels, self._action_params['undefined_action_id'])
            valid_labels = tf.boolean_mask(matched_labels, valid_action_mask)
            valid_logits = tf.boolean_mask(matches_logits, valid_action_mask)
            valid_embeddings = tf.boolean_mask(matches_embeddings, valid_action_mask)
            num_valid_labels = tf.size(valid_labels)

            # add ops to update moving action centers
            self._update_moving_centers(valid_embeddings, valid_labels)

            # scheduled scale for logits
            scale = self._get_action_scale(self._action_params['scale_start'], self._action_params['scale_end'],
                                           self._action_params['scale_num_steps'], self._action_params['scale_power'])
            # scale = self._get_action_scale(self._action_params['scale_start'], valid_labels, valid_logits)
            tf.add_to_collection('accuracy_summary', tf.summary.scalar('action_scale', scale))

            # ce loss for valid action classes
            action_ce_loss_value = weighted_ce_loss(
                valid_labels, scale * valid_logits, self._action_params['num_actions'], 'ce_loss',
                self._action_params['max_entropy_weight'], limits=self._action_params['weight_limits'],
                alpha=self._action_params['focal_alpha'], gamma=self._action_params['focal_gamma'],
                num_bins=self._action_params['num_bins'])

            # global pull-push loss for valid action classes with moving centers
            glob_pull_push_loss_value = tf.cond(
                tf.equal(num_valid_labels, 0),
                lambda: 0.0,
                lambda: explicit_pull_push_loss(valid_embeddings, valid_labels, self._action_moving_centers,
                                                self._action_params['glob_pull_push_margin'], 'glob_pull_push_loss',
                                                self._action_params['glob_pull_push_loss_top_k']))

            # center loss for valid action classes with moving centers
            center_loss_value = tf.cond(tf.equal(num_valid_labels, 0),
                                        lambda: 0.0,
                                        lambda: explicit_center_loss(valid_embeddings, valid_labels,
                                                                     self._action_moving_centers, 'glob_pull_push_loss',
                                                                     self._action_params['center_loss_top_k']))

            # local push loss for valid action classes
            local_push_loss_value = tf.cond(tf.equal(num_valid_labels, 0),
                                            lambda: 0.0,
                                            lambda: local_push_loss(valid_embeddings, valid_labels,
                                                                    self._action_params['num_actions'],
                                                                    self._action_params['local_push_margin'],
                                                                    'local_push_loss',
                                                                    self._action_params['local_push_top_k']))

            # sampling loss for valid action classes
            sampling_loss_value = tf.cond(tf.equal(num_valid_labels, 0),
                                          lambda: 0.0,
                                          lambda: sampling_losses(valid_embeddings, valid_labels,
                                                                  self._action_moving_centers,
                                                                  self._action_params['num_samples'],
                                                                  self._action_params['ce_loss_weight'],
                                                                  self._action_params['auxiliary_loss_weight'],
                                                                  'sampling_loss', scale,
                                                                  self._action_params['max_entropy_weight'],
                                                                  limits=self._action_params['weight_limits'],
                                                                  alpha=self._action_params['focal_alpha'],
                                                                  gamma=self._action_params['focal_gamma']))

            return action_ce_loss_value, glob_pull_push_loss_value, center_loss_value, local_push_loss_value,\
                   sampling_loss_value

    def _build_losses(self, labels, annot, backbone_name):
        """Adds losses to the training graph.

        :param labels: BBox labels
        :param annot: BBox coordinates
        :param backbone_name: Target backbone name
        """

        with tf.name_scope(self._name + '_losses'):
            # Detection losses
            det_conf_loss, det_loc_loss, matches_mask, matched_scores, gt_sample_ids, gt_labels =\
                self._multibox_loss(labels, annot, class_agnostic=True, output_original_labels=True)
            self._add_loss_summary(det_conf_loss, 'conf_loss')
            self._add_loss_summary(det_loc_loss, 'loc_loss')
            total_detection_loss = adaptive_weighting([det_conf_loss, det_loc_loss], name='detection_loss')
            self._add_loss_summary(total_detection_loss, 'total_detection_loss')

            # Action losses
            action_ce_loss_value, glob_pull_push_loss_value, center_loss_value, local_push_loss_value, \
            sampling_loss_value =\
                self._action_losses(matches_mask, matched_scores, gt_sample_ids, gt_labels)
            self._add_loss_summary(action_ce_loss_value, 'action_ce_loss')
            self._add_loss_summary(glob_pull_push_loss_value, 'glob_pull_push_loss')
            self._add_loss_summary(center_loss_value, 'center_loss')
            self._add_loss_summary(local_push_loss_value, 'local_push_loss_value')
            self._add_loss_summary(sampling_loss_value, 'sampling_loss')

            # Total Action loss
            auxiliary_action_loss = adaptive_weighting(
                [glob_pull_push_loss_value, center_loss_value, local_push_loss_value, sampling_loss_value],
                name='auxiliary_action_loss')
            self._add_loss_summary(auxiliary_action_loss, 'auxiliary_action_loss')

            total_action_loss = tf.add_n([self._action_params['ce_loss_weight'] * action_ce_loss_value,
                                          self._action_params['auxiliary_loss_weight'] * auxiliary_action_loss],
                                         name='total_action_loss')
            self._add_loss_summary(total_action_loss, 'total_action_loss')

            # Total loss
            main_loss = adaptive_weighting([total_detection_loss, total_action_loss], name='main_loss')
            self._add_loss_summary(main_loss, 'main_loss')
            total_loss = tf.identity(main_loss, name='total_loss')
            self._add_loss_summary(total_loss, 'total_loss')
            self._model['total_loss'] = total_loss

    @property
    def predictions(self):
        """Returns model predictions Op

        :return: Predicted locations, confidences and action classes Ops
        """

        return self._model['pr_loc'], self._model['pr_det_conf'], self._model['pr_action_conf']
