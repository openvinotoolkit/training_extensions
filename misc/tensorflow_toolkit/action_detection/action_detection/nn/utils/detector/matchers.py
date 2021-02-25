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

from action_detection.nn.nodes.metrics import iou_similarity


def ssd_match(gt_data, valid_gt_mask, priors_data, threshold, output_second_order=False):
    """Carry out SSD-like two stage matching between sets of predicted and ground truth boxes.

    :param gt_data: Ground truth set of boxes
    :param valid_gt_mask: Mask of valid to match ground truth boxes
    :param priors_data: Set of anchor boxes
    :param threshold: Min IoU threshold to match boxes
    :param output_second_order: Whether to output top-2 matches
    :return: ID and IoU score of matched ground truth box or -1 for each prior box
    """

    def _match_gt_to_priors(sim_matrix):
        """Carry out first matching stage: ground truth to priors.

        :param sim_matrix: Similarity matrix
        :return: Tuple of matched IDs and IoU scores
        """

        batch = tf.shape(gt_data)[0]
        num_gt = tf.shape(gt_data)[1]
        num_priors = tf.shape(priors_data)[0]

        best_priors_local_ids = tf.reshape(tf.argmax(sim_matrix, axis=1), [batch, -1])
        best_priors_scores = tf.reshape(tf.reduce_max(sim_matrix, axis=1), [-1])

        best_priors_glob_ids = tf.reshape(
            tf.cast(best_priors_local_ids, tf.int32) +
            tf.reshape(tf.range(0, batch * num_priors, num_priors, dtype=tf.int32), [batch, 1]), [-1])

        mask = tf.logical_and(tf.reshape(valid_gt_mask, [-1]), tf.greater(best_priors_scores, 0.0))
        valid_priors_glob_ids = tf.boolean_mask(tf.cast(best_priors_glob_ids, tf.int64), mask)
        valid_gt_glob_ids = tf.boolean_mask(tf.range(0, batch * num_gt, dtype=tf.int32), mask)
        valid_matched_scores = tf.boolean_mask(best_priors_scores, mask)

        sparse_indices = tf.expand_dims(valid_priors_glob_ids, 1)

        sparse_matched_ids = tf.SparseTensor(sparse_indices, valid_gt_glob_ids, [batch * num_priors])
        matched_ids = tf.reshape(
            tf.sparse_tensor_to_dense(sparse_matched_ids, default_value=-1, validate_indices=False),
            [batch, num_priors])
        sparse_matched_scores = tf.SparseTensor(sparse_indices, valid_matched_scores, [batch * num_priors])

        matched_scores = tf.reshape(
            tf.sparse_tensor_to_dense(sparse_matched_scores, default_value=0.0, validate_indices=False),
            [batch, num_priors])

        return matched_ids, matched_scores

    def _match_priors_to_gt(sim_matrix):
        """Carry out second matching stage: priors to ground truth

        :param sim_matrix: Similarity matrix
        :return: Tuple of matched IDs and IoU scores
        """

        batch = tf.shape(gt_data)[0]
        num_gt = tf.shape(gt_data)[1]
        num_priors = tf.shape(priors_data)[0]

        if output_second_order:
            sim_matrix = tf.transpose(tf.reshape(sim_matrix, [batch, num_gt, num_priors]), [0, 2, 1])
            top_gt_scores, top_gt_local_ids = tf.nn.top_k(sim_matrix, 2, sorted=True)
        else:
            sim_matrix = tf.reshape(sim_matrix, [batch, num_gt, num_priors])
            top_gt_local_ids = tf.argmax(sim_matrix, axis=1)
            top_gt_scores = tf.reduce_max(sim_matrix, axis=1)

        return top_gt_local_ids, top_gt_scores

    def _translate_matched_priors_to_gt(best_gt_local_ids, best_gt_scores):
        """Converts matched IDs and score from internal format into output.

        :param best_gt_local_ids: matched IDs
        :param best_gt_scores: matched IoU scores
        :return: Tuple of matched IDs and IoU scores
        """

        batch = tf.shape(gt_data)[0]
        num_gt = tf.shape(gt_data)[1]

        best_gt_glob_ids = tf.cast(best_gt_local_ids, tf.int32) + \
                           tf.reshape(tf.range(start=0, limit=batch * num_gt, delta=num_gt, dtype=tf.int32), [batch, 1])

        valid_matches_mask = tf.greater(best_gt_scores, threshold)
        matched_ids = tf.where(valid_matches_mask,
                               tf.cast(best_gt_glob_ids, tf.int32),
                               tf.fill(tf.shape(best_gt_glob_ids), -1))
        matched_scores = tf.where(valid_matches_mask,
                                  best_gt_scores,
                                  tf.zeros_like(best_gt_scores))

        return matched_ids, matched_scores

    similarity_matrix = iou_similarity(tf.reshape(gt_data, [-1, 4]), priors_data)
    similarity_matrix = tf.where(tf.reshape(valid_gt_mask, [-1]),
                                 similarity_matrix,
                                 tf.zeros_like(similarity_matrix))

    gt_to_priors_matches, gt_to_priors_scores = _match_gt_to_priors(similarity_matrix)
    top_priors_to_gt_matches, top_priors_to_gt_scores = _match_priors_to_gt(similarity_matrix)

    if output_second_order:
        best_priors_to_gt_matches, second_priors_to_gt_matches = tf.unstack(top_priors_to_gt_matches, axis=-1)
        best_priors_to_gt_scores, second_priors_to_gt_scores = tf.unstack(top_priors_to_gt_scores, axis=-1)
    else:
        best_priors_to_gt_matches = top_priors_to_gt_matches
        best_priors_to_gt_scores = top_priors_to_gt_scores

    best_priors_to_gt_matches, best_priors_to_gt_scores = _translate_matched_priors_to_gt(
        best_priors_to_gt_matches, best_priors_to_gt_scores)

    gt_to_priors_matches_mask = tf.greater_equal(gt_to_priors_matches, 0)
    best_matched_ids = tf.where(gt_to_priors_matches_mask, gt_to_priors_matches, best_priors_to_gt_matches)
    best_matched_scores = tf.where(gt_to_priors_matches_mask, gt_to_priors_scores, best_priors_to_gt_scores)

    if output_second_order:
        # noinspection PyUnboundLocalVariable
        second_priors_to_gt_matches, second_priors_to_gt_scores = _translate_matched_priors_to_gt(
            second_priors_to_gt_matches, second_priors_to_gt_scores)

        return best_matched_ids, best_matched_scores, second_priors_to_gt_matches, second_priors_to_gt_scores
    else:
        return best_matched_ids, best_matched_scores, None, None
