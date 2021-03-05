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

from collections import namedtuple

import tensorflow as tf

from action_detection.nn.utils.detector.bounding_boxes import center_encode, center_decode
from action_detection.nn.utils.detector.matchers import ssd_match

from action_detection.nn.backbones import get_backbone, get_orthogonal_scope_name
from action_detection.nn.models.base_network import BaseNetwork
from action_detection.nn.nodes.losses import weight_decay, unit_gamma, orthogonal_conv, focal_ce_loss,\
    adaptive_weighting, balanced_l1_loss, gradient_harmonized_ce_loss, max_entropy_ce_loss
from action_detection.nn.nodes.ops import conv2d, safe_reduce_op
from action_detection.nn.utils.detector.priors import generate_clustered_prior_boxes

SSDHeadDesc = namedtuple('SSDHeadDesc', 'scale, anchors, num_classes, clip, offset, internal_size')
SSDHead = namedtuple('SSDHead', 'name, loc, conf, priors, num_classes, num_priors')


class SSD(BaseNetwork):
    """Describes network for the general object detection (OD) problem.
    """

    def __init__(self, backbone_name, net_input, labels, annot, fn_activation, is_training, head_params, merge_bn,
                 merge_bn_transition, lr_params, mbox_param, wd_weight=1e-2, global_step=None, name='ssd',
                 use_nesterov=True, norm_kernels=False):
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
        :param mbox_param: Parameters for training loss
        :param wd_weight: Weight decay value
        :param global_step: Variable for counting the training steps if exists
        :param name: Network name
        :param use_nesterov: Whether to enable nesterov momentum calculation
        :param norm_kernels: Whether to normalize convolution kernels
        """

        super(SSD, self).__init__(is_training, merge_bn, merge_bn_transition, lr_params, wd_weight, global_step,
                                  use_nesterov, norm_kernels)

        self._fn_activation = fn_activation
        self._head_params = head_params
        self._name = name
        self._mbox_param = mbox_param

        self._model['input'] = net_input
        self._model['labels'] = labels
        self._model['annot'] = annot
        self._model['batch_size'] = tf.shape(net_input)[0]

        self._build_network(net_input, backbone_name)
        if is_training is not None:
            self._create_lr_schedule()
            self._build_losses(labels, annot, backbone_name)

    def _add_detection_head(self, input_value, input_size, internal_size, num_classes, num_anchors, name):
        """Adds single SSD head on top of the specified input features

        :param input_value: Input features
        :param input_size: Number of input channels
        :param internal_size: Number of channels for the internal representation
        :param num_classes: Number of taget classes
        :param num_anchors: Number of anchor boxes
        :param name: Name of block
        :return: Location and confidence tensors
        """

        with tf.variable_scope(name):
            loc_conv1 = conv2d(input_value, [1, 1, input_size, internal_size], 'loc_conv1',
                               use_bias=False, use_bn=True, is_training=self._is_training,
                               fn_activation=self._fn_activation, norm_kernel=self._norm_kernels,
                               merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition)
            loc_conv2 = conv2d(loc_conv1, [3, 3, internal_size, 1], 'loc_conv2',
                               depth_wise=True, use_bias=False, use_bn=True, is_training=self._is_training,
                               merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                               norm_kernel=self._norm_kernels)
            locations = conv2d(loc_conv2, [1, 1, internal_size, num_anchors * 4], 'loc',
                               use_bias=False, use_bn=True, is_training=self._is_training,
                               merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                               pr_product=True, add_summary=True, norm_kernel=self._norm_kernels)

            conf_conv1 = conv2d(input_value, [1, 1, input_size, internal_size], 'conf_conv1',
                                use_bias=False, use_bn=True, is_training=self._is_training,
                                fn_activation=self._fn_activation, norm_kernel=self._norm_kernels,
                                merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition)
            conf_conv2 = conv2d(conf_conv1, [3, 3, internal_size, 1], 'conf_conv2',
                                depth_wise=True, use_bias=False, use_bn=True, is_training=self._is_training,
                                merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                                norm_kernel=self._norm_kernels)
            confidences = conv2d(conf_conv2, [1, 1, internal_size, num_anchors * num_classes], 'conf',
                                 use_bias=False, use_bn=True, is_training=self._is_training,
                                 merge_op=self._merge_bn, merge_op_transit=self._merge_bn_transition,
                                 pr_product=True, add_summary=True, norm_kernel=self._norm_kernels)

        return locations, confidences

    def _build_detection_heads(self, head_params, skeleton, prefix='output_'):
        """Creates all detection heads according parameters.

        :param head_params: Parmaters of heads
        :param skeleton: Network skeleton to add heads
        :param prefix: Name prefix for the new created heads
        """

        with tf.variable_scope('detection_heads'):
            heads = []
            all_loc = []
            all_conf = []
            all_priors = []
            total_num_priors = 0

            for i, head_param in enumerate(head_params):
                place_name = '{}{}x'.format(prefix, head_param.scale)
                network_input = skeleton[place_name]
                anchors = head_param.anchors

                loc, conf = self._add_detection_head(
                    network_input, network_input.get_shape()[3], head_param.internal_size,
                    head_param.num_classes, len(anchors), 'head_{}'.format(i + 1))
                all_loc.append(tf.reshape(loc, [self._model['batch_size'], -1, 4]))
                all_conf.append(tf.reshape(conf, [self._model['batch_size'], -1, head_param.num_classes]))

                feature_size = network_input.get_shape()[1:3]
                image_size = self._model['input'].get_shape().as_list()[1:3]
                priors_array, num_priors =\
                    generate_clustered_prior_boxes(feature_size, image_size, anchors, head_param.scale,
                                                   head_param.clip, head_param.offset)
                priors = tf.constant(priors_array, name='head_{}/PriorBoxClustered'.format(i))

                all_priors.append(priors)
                total_num_priors += num_priors

                heads.append(SSDHead(name=place_name, loc=loc, conf=conf, priors=priors,
                                     num_classes=head_param.num_classes, num_priors=num_priors))

                tf.add_to_collection('activation_summary', tf.summary.histogram(place_name + '_loc', loc))
                tf.add_to_collection('activation_summary', tf.summary.histogram(place_name + '_conf', conf))

        self._model['heads'] = heads
        self._model['encoded_loc'] = tf.concat(all_loc, axis=1, name='out_detection_loc')
        self._model['logits'] = tf.concat(all_conf, axis=1, name='out_detection_logits')
        self._model['priors'] = tf.concat(all_priors, axis=0, name='out_detection_priors')
        self._model['total_num_priors'] = total_num_priors

        self._model['pr_loc'] = center_decode(self._model['encoded_loc'], self._model['priors'],
                                              self._mbox_param['variance'], clip=self._is_training is None)
        self._model['pr_det_conf'] = tf.nn.softmax(self._model['logits'], axis=-1, name='out_detection_conf')

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

            self._build_detection_heads(self._head_params, skeleton)

    @staticmethod
    def _get_mean_matched_values(trg_ids, src_values, name=None):
        """Calculates mean bbox for each instance.

        :param trg_ids: Ground truth IDs
        :param src_values: Values to average
        :param name: Name of block
        :return: GT IDs and averaged bboxes tuple
        """

        def _process():
            unique_gt_ids, unique_gt_ids_idx, unique_gt_ids_counts = tf.unique_with_counts(trg_ids)

            dense_shape = [tf.size(unique_gt_ids), tf.size(trg_ids)]
            dense_size = dense_shape[0] * dense_shape[1]
            range_ids = tf.range(0, tf.size(trg_ids), dtype=tf.int32)
            grouped_sparse_idx = tf.expand_dims(tf.cast(unique_gt_ids_idx * tf.size(trg_ids) + range_ids, tf.int64), 1)

            normalizer = tf.reciprocal(tf.cast(unique_gt_ids_counts, tf.float32))
            unpacked_values = tf.unstack(src_values, axis=-1)

            mean_slices = []
            for values_slice in unpacked_values:
                dense_values_slice = tf.reshape(tf.sparse_tensor_to_dense(
                    tf.SparseTensor(grouped_sparse_idx, values_slice, [dense_size]),
                    default_value=0.0, validate_indices=False), dense_shape)
                mean_slice = normalizer * tf.reduce_sum(dense_values_slice, axis=-1)
                mean_slices.append(tf.reshape(mean_slice, [-1, 1]))

            return unique_gt_ids, tf.concat(tuple(mean_slices), axis=1)

        with tf.name_scope(name, 'mean_matched_loc'):
            return tf.cond(tf.greater(tf.size(trg_ids), 0),
                           lambda: _process(),
                           lambda: (trg_ids, src_values))

    def _get_top_matched_mask(self, gt_ids, matched_scores, threshold, max_num_samples_per_gt, name,
                              drop_ratio=None, min_default_score=1e-3, soft_num_samples=True):
        """Filters top-k matched prior boxes per instance.

        :param gt_ids: Ground truth IDs
        :param matched_scores: Score of matched priors
        :param threshold: Threshold to filter low-confidence matches
        :param max_num_samples_per_gt: Size of top-k queue
        :param name: Name of block
        :param drop_ratio: Ratio of matches per instance to drop
        :param min_default_score: Min possible score
        :param soft_num_samples: Whether to relax hard top-k border into soft one
        :return: mask of filtered matches and per instance weights
        """

        assert threshold >= 0.0
        assert max_num_samples_per_gt > 0
        assert min_default_score > 0.0

        def _process():
            unique_gt_ids, unique_gt_ids_idx, unique_gt_ids_counts = tf.unique_with_counts(gt_ids)
            dense_shape = [tf.size(unique_gt_ids), tf.size(gt_ids)]
            dense_size = dense_shape[0] * dense_shape[1]
            range_ids = tf.range(0, tf.size(gt_ids), dtype=tf.int32)
            grouped_sparse_idx = tf.expand_dims(
                tf.cast(unique_gt_ids_idx * tf.size(gt_ids) + range_ids, tf.int64), 1)

            if drop_ratio is not None:
                drop_mask = tf.less(tf.random_uniform([tf.size(matched_scores)], 0.0, 1.0), drop_ratio)
                dropped_matched_scores = tf.where(
                    drop_mask, tf.fill(tf.shape(matched_scores), min_default_score), matched_scores)
            else:
                dropped_matched_scores = matched_scores

            grouped_matched_scores = tf.sparse_tensor_to_dense(
                tf.SparseTensor(grouped_sparse_idx, dropped_matched_scores, [dense_size]),
                default_value=0.0, validate_indices=False)
            grouped_positions = tf.sparse_tensor_to_dense(
                tf.SparseTensor(grouped_sparse_idx, range_ids, [dense_size]),
                default_value=-1, validate_indices=False)

            if soft_num_samples:
                num_samples_per_gt = tf.maximum(max_num_samples_per_gt, tf.reduce_min(unique_gt_ids_counts))
                tf.add_to_collection('loss_summary', tf.summary.scalar('top_positives/num_per_gt', num_samples_per_gt))
            else:
                num_samples_per_gt = max_num_samples_per_gt

            top_values, top_indices = tf.nn.top_k(
                tf.reshape(grouped_matched_scores, dense_shape), k=num_samples_per_gt, sorted=False)
            valid_top_values_mask = tf.greater(top_values, threshold)

            shifted_top_indices = top_indices + tf.reshape(
                tf.range(0, tf.size(unique_gt_ids), dtype=tf.int32) * tf.size(gt_ids), [-1, 1])
            filtered_top_indices = tf.boolean_mask(shifted_top_indices, valid_top_values_mask)
            out_ids = tf.gather(grouped_positions, filtered_top_indices)
            self._add_loss_summary(tf.size(out_ids), 'num_matches')

            num_valid_per_match = tf.reduce_sum(tf.cast(valid_top_values_mask, tf.float32), axis=-1, keepdims=True)
            tiled_num_valid_per_match = tf.tile(num_valid_per_match, [1, num_samples_per_gt])
            out_num_valid_per_match = tf.boolean_mask(tiled_num_valid_per_match, valid_top_values_mask)
            out_instance_weights = tf.reciprocal(tf.maximum(1.0, out_num_valid_per_match))

            sparse_to_dens_ids = tf.expand_dims(tf.cast(out_ids, tf.int64), 1)
            out_matched_mask = tf.sparse_tensor_to_dense(
                tf.SparseTensor(sparse_to_dens_ids, tf.ones([tf.size(out_ids)], dtype=tf.bool), [tf.size(gt_ids)]),
                default_value=False, validate_indices=False)

            return out_matched_mask, out_instance_weights

        with tf.name_scope(name):
            return tf.cond(tf.greater(tf.size(gt_ids), 0),
                           lambda: _process(),
                           lambda: (tf.zeros_like(gt_ids, dtype=tf.bool), tf.ones_like(gt_ids, dtype=tf.float32)))

    def _multibox_loss(self, gt_labels, gt_bboxes, class_agnostic=False, output_original_labels=False):
        """Creates main detector losses.

        :param gt_labels: BBox labels
        :param gt_bboxes: BBox coordinates
        :param class_agnostic: Whether to interpret positive labels as single positive class
        :param output_original_labels: Map of original class labels
        :return: Tuple of confidence and location losses
        """

        def _default_loc_loss(pr_encoded_loc, gt_encoded_loc):
            """Location loss

            :param pr_encoded_loc: Encoded predicted locations
            :param gt_encoded_loc: Encoded ground truth locations
            :return: Loss value
            """

            with tf.name_scope('loc_loss'):
                diff = pr_encoded_loc - gt_encoded_loc
                loss_values = balanced_l1_loss(diff, alpha=0.5, gamma=1.5)
                return tf.reduce_sum(loss_values, axis=1)

        def _conf_loss(predicted_conf, gt_classes, name, weights=None):
            """Confidence loss

            :param predicted_conf: Predicted class distributions
            :param gt_classes: Ground truth classes
            :param name: Name of block
            :param weights: Instance weights
            :return: Loss value
            """

            enable_max_entropy_loss = 'entropy_weight' in self._mbox_param and self._mbox_param['entropy_weight'] > 0.0
            enable_focal_loss = 'focal_alpha' in self._mbox_param and self._mbox_param['focal_alpha'] > 0.0 and \
                                'focal_gamma' in self._mbox_param and self._mbox_param['focal_gamma'] > 0.0
            enable_gh_loss = 'gh_num_bins' in self._mbox_param and self._mbox_param['gh_num_bins'] > 0
            if enable_max_entropy_loss and enable_focal_loss and enable_gh_loss:
                raise Exception('Cannot enable different CE losses simultaneously')

            with tf.variable_scope(name):
                if enable_max_entropy_loss:
                    ce_losses = max_entropy_ce_loss(gt_classes, predicted_conf,
                                                    self._mbox_param['entropy_weight'], 'conf_ce_losses',
                                                    enable_gsoftmax=False)
                elif enable_focal_loss:
                    ce_losses = focal_ce_loss(gt_classes, predicted_conf,
                                              self._mbox_param['focal_alpha'], self._mbox_param['focal_gamma'],
                                              'conf_ce_losses')
                elif enable_gh_loss:
                    ce_losses = gradient_harmonized_ce_loss(gt_classes, predicted_conf, 'conf_ce_losses',
                                                            self._mbox_param['gh_num_bins'])
                else:
                    ce_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_classes, logits=predicted_conf,
                                                                               name='ce_losses')

                if weights is not None:
                    ce_losses = ce_losses * weights
                ce_losses = tf.boolean_mask(ce_losses, tf.greater(ce_losses, 0.0))
            return ce_losses

        def _get_hard_samples(values, out_num):
            """Safely extracts top-k values.

            :param values: Input values
            :param out_num: Number of values to extract
            :return: Extracted values
            """

            def _hardest():
                top_values, _ = tf.nn.top_k(values, out_num)
                return top_values

            with tf.name_scope('hard_sample_mining'):
                num_values = tf.size(values)
                return tf.cond(tf.less_equal(num_values, out_num), lambda: values, _hardest)

        def _giou_loss(pr_bboxes, input_gt_bboxes):
            """Calculates General IoU loss directly over bboxes.

            :param pr_bboxes: Predicted bbox coordinates
            :param input_gt_bboxes: Ground truth bbox coordinates
            :return: Loss value
            """

            with tf.name_scope('giou_loss'):
                fixed_pr_ymin = tf.minimum(pr_bboxes[:, 0], pr_bboxes[:, 2])
                fixed_pr_xmin = tf.minimum(pr_bboxes[:, 1], pr_bboxes[:, 3])
                fixed_pr_ymax = tf.maximum(pr_bboxes[:, 0], pr_bboxes[:, 2])
                fixed_pr_xmax = tf.maximum(pr_bboxes[:, 1], pr_bboxes[:, 3])

                intersect_ymin = tf.maximum(fixed_pr_ymin, input_gt_bboxes[:, 0])
                intersect_xmin = tf.maximum(fixed_pr_xmin, input_gt_bboxes[:, 1])
                intersect_ymax = tf.minimum(fixed_pr_ymax, input_gt_bboxes[:, 2])
                intersect_xmax = tf.minimum(fixed_pr_xmax, input_gt_bboxes[:, 3])

                intersect_height = tf.maximum(0.0, intersect_ymax - intersect_ymin)
                intersect_width = tf.maximum(0.0, intersect_xmax - intersect_xmin)
                intersect_areas = intersect_width * intersect_height

                pr_areas = (fixed_pr_ymax - fixed_pr_ymin) * (fixed_pr_xmax - fixed_pr_xmin)
                gt_areas = (input_gt_bboxes[:, 2] - input_gt_bboxes[:, 0]) *\
                           (input_gt_bboxes[:, 3] - input_gt_bboxes[:, 1])
                union_areas = pr_areas + gt_areas - intersect_areas

                overlaps = tf.where(tf.greater(union_areas, 0.0),
                                    intersect_areas / union_areas,
                                    tf.zeros_like(intersect_areas))

                enclose_ymin = tf.minimum(fixed_pr_ymin, input_gt_bboxes[:, 0])
                enclose_xmin = tf.minimum(fixed_pr_xmin, input_gt_bboxes[:, 1])
                enclose_ymax = tf.maximum(fixed_pr_ymax, input_gt_bboxes[:, 2])
                enclose_xmax = tf.maximum(fixed_pr_xmax, input_gt_bboxes[:, 3])

                enclose_height = tf.maximum(0.0, enclose_ymax - enclose_ymin)
                enclose_width = tf.maximum(0.0, enclose_xmax - enclose_xmin)
                enclose_areas = enclose_width * enclose_height

                enclose_ratio = tf.where(tf.greater(enclose_areas, 0.0),
                                         (enclose_areas - union_areas) / enclose_areas,
                                         tf.zeros_like(enclose_areas))
                generalized_overlaps = overlaps - enclose_ratio
                out_losses = 1.0 - generalized_overlaps

                return out_losses

        def _repulsion_loss(pr_bboxes, first_gt_ids, second_gt_ids, scores, eps=1e-5):
            """Calculates Repulsion loss between current predicted bbox and top-2 anchor box.

            :param pr_bboxes: Predicted bbox coordinates
            :param first_gt_ids: Coordinates of matched GT bboxes
            :param second_gt_ids: Coordinates of closest GT bboxes
            :param scores: Matched scores
            :param eps: Epsilon scalar value
            :return: Loss value
            """

            def _process(mask):
                valid_pr_bboxes = tf.boolean_mask(pr_bboxes, mask)
                valid_first_gt_ids = tf.boolean_mask(first_gt_ids, mask)
                valid_second_gt_ids = tf.boolean_mask(second_gt_ids, mask)

                first_gt_bboxes = tf.gather(tf.reshape(gt_bboxes, [-1, 4]), valid_first_gt_ids)
                second_gt_bboxes = tf.gather(tf.reshape(gt_bboxes, [-1, 4]), valid_second_gt_ids)

                fixed_pr_ymin = tf.minimum(valid_pr_bboxes[:, 0], valid_pr_bboxes[:, 2])
                fixed_pr_xmin = tf.minimum(valid_pr_bboxes[:, 1], valid_pr_bboxes[:, 3])
                fixed_pr_ymax = tf.maximum(valid_pr_bboxes[:, 0], valid_pr_bboxes[:, 2])
                fixed_pr_xmax = tf.maximum(valid_pr_bboxes[:, 1], valid_pr_bboxes[:, 3])

                trg_intersect_ymin = tf.maximum(fixed_pr_ymin, second_gt_bboxes[:, 0])
                trg_intersect_xmin = tf.maximum(fixed_pr_xmin, second_gt_bboxes[:, 1])
                trg_intersect_ymax = tf.minimum(fixed_pr_ymax, second_gt_bboxes[:, 2])
                trg_intersect_xmax = tf.minimum(fixed_pr_xmax, second_gt_bboxes[:, 3])

                gt_intersect_ymin = tf.maximum(first_gt_bboxes[:, 0], second_gt_bboxes[:, 0])
                gt_intersect_xmin = tf.maximum(first_gt_bboxes[:, 1], second_gt_bboxes[:, 1])
                gt_intersect_ymax = tf.minimum(first_gt_bboxes[:, 2], second_gt_bboxes[:, 2])
                gt_intersect_xmax = tf.minimum(first_gt_bboxes[:, 3], second_gt_bboxes[:, 3])

                trg_intersect_areas = tf.maximum(0.0, trg_intersect_ymax - trg_intersect_ymin) *\
                                      tf.maximum(0.0, trg_intersect_xmax - trg_intersect_xmin)
                gt_intersect_areas = tf.maximum(0.0, gt_intersect_ymax - gt_intersect_ymin) *\
                                     tf.maximum(0.0, gt_intersect_xmax - gt_intersect_xmin)
                gt_areas = (second_gt_bboxes[:, 2] - second_gt_bboxes[:, 0]) * \
                           (second_gt_bboxes[:, 3] - second_gt_bboxes[:, 1])

                shifted_intersect_over_gt = tf.where(tf.greater(gt_areas, 0.0),
                                                     (trg_intersect_areas - gt_intersect_areas) / gt_areas,
                                                     tf.zeros_like(trg_intersect_areas))

                valid_intersections_mask = tf.greater(shifted_intersect_over_gt, 0.0)
                valid_intersections = tf.boolean_mask(shifted_intersect_over_gt, valid_intersections_mask)
                out_losses = tf.negative(tf.log(1.0 + eps - valid_intersections))

                return out_losses

            with tf.name_scope('repulsion_loss'):
                valid_mask = tf.greater(scores, 0.0)
                num_valid_pairs = tf.reduce_sum(tf.cast(valid_mask, tf.int32))

                return tf.cond(tf.greater(num_valid_pairs, 0),
                               lambda: _process(valid_mask),
                               lambda: tf.zeros([0], dtype=tf.float32))

        def _compactness_loss(src_gt_idx, src_decoded_loc, src_gt_bboxes):
            """Calculates location loss between GT and mean bbox coordinates.

            :param src_gt_idx: Ground truth IDs
            :param src_decoded_loc: Decoded predicted locations
            :param src_gt_bboxes: Ground truth bbox coordinates
            :return: Loss value
            """

            with tf.name_scope('compactness_loss'):
                trg_gt_idx, mean_decoded_loc = self._get_mean_matched_values(src_gt_idx, src_decoded_loc)
                trg_gt_bboxes = tf.gather(tf.reshape(src_gt_bboxes, [-1, 4]), trg_gt_idx)
                return _giou_loss(mean_decoded_loc, trg_gt_bboxes)

        with tf.name_scope('multibox_loss'):
            enable_comp_loss =\
                'comp_loss_max_num_samples' in self._mbox_param and self._mbox_param['comp_loss_max_num_samples'] > 0
            enable_repulsion_loss = enable_comp_loss and self._mbox_param['repulsion_loss_weight'] > 0.0

            valid_data_mask = tf.greater_equal(gt_labels, 0)
            matches_ids, matched_scores, second_matches_ids, second_matched_scores = ssd_match(
                gt_bboxes, valid_data_mask, self._model['priors'], self._mbox_param['threshold'],
                output_second_order=enable_repulsion_loss)
            matches_mask = tf.greater_equal(matches_ids, 0)

            valid_gt_ids = tf.boolean_mask(matches_ids, matches_mask)
            valid_matched_scores = tf.boolean_mask(matched_scores, matches_mask)

            if enable_repulsion_loss:
                second_valid_gt_ids = tf.boolean_mask(second_matches_ids, matches_mask)
                second_valid_matched_scores = tf.boolean_mask(second_matched_scores, matches_mask)

            all_prior_ids = tf.tile(tf.reshape(tf.range(0, tf.shape(matches_ids)[1], dtype=tf.int32), [1, -1]),
                                    [tf.shape(matches_ids)[0], 1])
            valid_prior_ids = tf.boolean_mask(all_prior_ids, matches_mask)

            positives_encoded_loc_data = tf.boolean_mask(self._model['encoded_loc'], matches_mask)
            positives_decoded_loc_data = tf.boolean_mask(self._model['pr_loc'], matches_mask)
            positives_conf_data = tf.boolean_mask(self._model['logits'], matches_mask)
            positives_gt_labels = tf.gather(tf.reshape(gt_labels, [-1]), valid_gt_ids)
            positives_gt_loc = tf.gather(tf.reshape(gt_bboxes, [-1, 4]), valid_gt_ids)
            positives_priors = tf.gather(self._model['priors'], valid_prior_ids)

            # Compactness location loss of extended set of matches
            if enable_comp_loss:
                best_matches_mask, _ = self._get_top_matched_mask(
                    valid_gt_ids, valid_matched_scores, 1e-7, self._mbox_param['comp_loss_max_num_samples'], 'det_top')
                self._add_loss_summary(tf.reduce_sum(tf.cast(best_matches_mask, tf.float32)) / tf.cast(
                    tf.maximum(1, tf.size(best_matches_mask)), tf.float32), 'compactness/density')

                auxiliary_pos_loc_data = tf.boolean_mask(positives_decoded_loc_data, best_matches_mask)
                auxiliary_valid_gt_ids = tf.boolean_mask(valid_gt_ids, best_matches_mask)

                comp_losses = _compactness_loss(auxiliary_valid_gt_ids, auxiliary_pos_loc_data, gt_bboxes)
                comp_loss = safe_reduce_op(comp_losses, tf.reduce_mean)
                self._add_loss_summary(comp_loss, 'comp_loss')

                if enable_repulsion_loss:
                    # noinspection PyUnboundLocalVariable
                    repulsion_losses = _repulsion_loss(auxiliary_pos_loc_data, auxiliary_valid_gt_ids,
                                                       tf.boolean_mask(second_valid_gt_ids, best_matches_mask),
                                                       tf.boolean_mask(second_valid_matched_scores, best_matches_mask))
                    repulsion_loss = safe_reduce_op(repulsion_losses, tf.reduce_mean)

            out_positives_gt_labels = positives_gt_labels

            do_instance_sampling =\
                'max_num_samples_per_gt' in self._mbox_param and self._mbox_param['max_num_samples_per_gt'] > 0
            if do_instance_sampling:
                best_matches_mask, best_matches_weights = self._get_top_matched_mask(
                    valid_gt_ids, valid_matched_scores, 1e-7, self._mbox_param['max_num_samples_per_gt'], 'det_top',
                    drop_ratio=self._mbox_param['matches_drop_ratio'])
                self._add_loss_summary(tf.reduce_sum(tf.cast(best_matches_mask, tf.float32)) / tf.cast(
                    tf.maximum(1, tf.size(best_matches_mask)), tf.float32), 'top_positives/density')

                positives_encoded_loc_data = tf.boolean_mask(positives_encoded_loc_data, best_matches_mask)
                positives_decoded_loc_data = tf.boolean_mask(positives_decoded_loc_data, best_matches_mask)
                positives_conf_data = tf.boolean_mask(positives_conf_data, best_matches_mask)
                positives_gt_labels = tf.boolean_mask(positives_gt_labels, best_matches_mask)
                positives_gt_loc = tf.boolean_mask(positives_gt_loc, best_matches_mask)
                positives_priors = tf.boolean_mask(positives_priors, best_matches_mask)

            # default L1 location loss
            encoded_gt_loc_data = center_encode(positives_gt_loc, positives_priors, self._mbox_param['variance'])
            positive_loc_losses = _default_loc_loss(positives_encoded_loc_data, encoded_gt_loc_data)
            default_loc_loss = safe_reduce_op(positive_loc_losses, tf.reduce_mean)
            self._model['loc_loss'] = default_loc_loss

            # GIoU location loss
            positive_giou_losses = _giou_loss(positives_decoded_loc_data, positives_gt_loc)
            giou_loss = safe_reduce_op(positive_giou_losses, tf.reduce_mean)
            self._model['giou_loss'] = giou_loss

            main_loc_loss_list = [default_loc_loss, giou_loss]
            if enable_comp_loss:
                # noinspection PyUnboundLocalVariable
                main_loc_loss_list.append(comp_loss)
            main_loc_loss = adaptive_weighting(main_loc_loss_list, name='gloc_loss')
            self._model['main_loc_loss'] = main_loc_loss
            self._add_loss_summary(main_loc_loss, 'main_loc_loss')

            if enable_repulsion_loss:
                # noinspection PyUnboundLocalVariable
                general_loc_loss = tf.add_n([main_loc_loss, self._mbox_param['repulsion_loss_weight'] * repulsion_loss],
                                            name='gloc_loss')
                self._model['general_loc_loss'] = general_loc_loss
                self._add_loss_summary(general_loc_loss, 'general_loc_loss')
                self._add_loss_summary(repulsion_loss, 'repulsion_loss')
            else:
                general_loc_loss = main_loc_loss

            fixed_positives_gt_labels = tf.ones_like(positives_gt_labels) if class_agnostic else positives_gt_labels
            # noinspection PyUnboundLocalVariable
            positive_conf_losses = _conf_loss(
                positives_conf_data, fixed_positives_gt_labels, 'pos_conf_loss',
                best_matches_weights if do_instance_sampling and self._mbox_param['instance_normalization'] else None)
            self._model['positive_conf_losses'] = positive_conf_losses

            negatives_conf_data = tf.boolean_mask(self._model['logits'], tf.logical_not(matches_mask))
            negatives_conf_losses =\
                _conf_loss(negatives_conf_data, tf.fill([tf.shape(negatives_conf_data)[0]],
                                                        self._mbox_param['bg_class']), 'neg_conf_loss')

            num_positives = tf.cast(tf.shape(positive_conf_losses)[0], tf.float32)
            num_negatives = tf.cast(float(self._mbox_param['neg_factor']) * num_positives, tf.int32)
            hard_negatives_conf_losses = _get_hard_samples(negatives_conf_losses, num_negatives)
            self._model['hard_negatives_conf_losses'] = hard_negatives_conf_losses

            num_conf_losses = tf.maximum(1, tf.size(positive_conf_losses) + tf.size(hard_negatives_conf_losses))
            conf_loss = tf.divide(tf.reduce_sum(positive_conf_losses) + tf.reduce_sum(hard_negatives_conf_losses),
                                  tf.cast(num_conf_losses, tf.float32))

        if output_original_labels:
            return conf_loss, general_loc_loss, matches_mask,\
                   valid_matched_scores, valid_gt_ids, out_positives_gt_labels
        else:
            return conf_loss, general_loc_loss

    def _build_losses(self, labels, annot, backbone_name):
        """Adds losses to the training graph.

        :param labels: BBox labels
        :param annot: BBox coordinates
        :param backbone_name: Target backbone name
        """

        with tf.name_scope(self._name + '_losses'):
            all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)

            conf_loss, loc_loss = self._multibox_loss(labels, annot)
            self._add_loss_summary(conf_loss, 'conf_loss')
            self._add_loss_summary(loc_loss, 'loc_loss')
            mbox_loss = self._mbox_param['cl_weight'] * conf_loss + loc_loss
            self._add_loss_summary(mbox_loss, 'mbox_loss')

            wd_loss = weight_decay(all_trainable_vars, self._weight_decay, 'var_reg')
            self._add_loss_summary(wd_loss, 'wd_loss')

            unit_gamma_loss = unit_gamma(all_trainable_vars, 0.1 * self._weight_decay, 'gamma_reg')
            self._add_loss_summary(unit_gamma_loss, 'gamma_loss')

            orthogonal_loss = orthogonal_conv(all_trainable_vars, 0.25 * self._model['lr'], 'ort_reg',
                                              key_scope=get_orthogonal_scope_name(backbone_name))
            self._add_loss_summary(orthogonal_loss, 'orthogonal_loss')

            total_loss = tf.add_n([mbox_loss, wd_loss, unit_gamma_loss, orthogonal_loss], name='total_loss')
            self._add_loss_summary(total_loss, 'total_loss')
            self._model['total_loss'] = total_loss

    @property
    def predictions(self):
        """Returns model predictions Op

        :return: Predicted locations and classes Ops
        """

        return self._model['pr_loc'], self._model['pr_det_conf']
