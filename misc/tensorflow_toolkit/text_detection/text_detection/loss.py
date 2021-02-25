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

""" This module contains loss definition. """

import tensorflow as tf


class ClassificationLoss():
    """ Classification loss for pixel-wise segmentation. """

    def __init__(self, conf):
        self.conf = conf

    @tf.function
    def __call__(self, groundtruth, segm_logits, sample_weight=None):
        segm_labels, segm_weights = groundtruth[:, :, :, :1], groundtruth[:, :, :, 1:2]
        segm_labels = tf.cast(segm_labels, tf.int32)

        def hard_negative_mining(scores, n_pos, neg_mask):
            if n_pos > 0:
                n_neg = n_pos * self.conf['max_neg_pos_ratio']
            else:
                n_neg = 10000

            max_neg_entries = tf.reduce_sum(input_tensor=tf.cast(neg_mask, tf.int32))

            n_neg = tf.minimum(n_neg, max_neg_entries)
            if n_neg > 0:
                neg_conf = tf.boolean_mask(tensor=scores, mask=neg_mask)
                vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
                threshold = vals[-1]
                selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
            else:
                selected_neg_mask = tf.zeros_like(neg_mask)

            return tf.cast(selected_neg_mask, tf.int32)

        def batch_hard_negative_mining(neg_conf, pos_mask, neg_mask):
            selected_neg_mask = []
            for image_idx in range(self.conf['batch_size']):
                image_neg_conf = neg_conf[image_idx, :]
                image_neg_mask = neg_mask[image_idx, :]
                image_pos_mask = pos_mask[image_idx, :]
                n_pos = tf.reduce_sum(input_tensor=tf.cast(image_pos_mask, tf.int32))
                selected_neg_mask.append(
                    hard_negative_mining(image_neg_conf, n_pos, image_neg_mask))

            selected_neg_mask = tf.stack(selected_neg_mask)
            return selected_neg_mask

        segm_labels_flatten = tf.reshape(segm_labels, [self.conf['batch_size'], -1])
        pos_segm_weights_flatten = tf.reshape(segm_weights, [self.conf['batch_size'], -1])

        pos_mask = tf.equal(segm_labels_flatten, self.conf['text_label'])
        neg_mask = tf.equal(segm_labels_flatten, self.conf['background_label'])

        n_pos = tf.reduce_sum(input_tensor=tf.cast(pos_mask, dtype=tf.float32))

        segm_logits_flatten = tf.reshape(segm_logits, [self.conf['batch_size'], -1,
                                                       self.conf['num_classes']])
        segm_scores_flatten = tf.nn.softmax(segm_logits_flatten)

        segm_loss = tf.keras.metrics.sparse_categorical_crossentropy(
            y_pred=segm_scores_flatten, y_true=tf.cast(pos_mask, dtype=tf.int32))

        segm_neg_scores = segm_scores_flatten[:, :, 0]
        selected_neg_pixel_mask = batch_hard_negative_mining(segm_neg_scores, pos_mask, neg_mask)
        segm_weights = pos_segm_weights_flatten + tf.cast(selected_neg_pixel_mask, tf.float32)
        n_neg = tf.cast(tf.reduce_sum(input_tensor=selected_neg_pixel_mask), tf.float32)

        loss = 0.0
        if n_neg + n_pos > 0:
            loss = tf.reduce_sum(input_tensor=segm_loss * segm_weights) / (n_neg + n_pos) * 2.0

        loss = loss / self.conf['num_replicas']

        return loss


class LinkageLoss():
    """ Classification loss for pixel-wise neighbourhood. """

    def __init__(self, conf):
        self.conf = conf

    @tf.function
    def __call__(self, groundtruth, link_logits, sample_weight=None):
        groundtruth = tf.squeeze(groundtruth, axis=-1)
        segm_labels, link_labels, link_weights = \
            groundtruth[:, :, :, :1], groundtruth[:, :, :, 2:10], groundtruth[:, :, :, 10:18]
        link_labels = tf.cast(link_labels, tf.int32)

        n_pos = tf.reduce_sum(tf.cast(tf.equal(segm_labels, self.conf['text_label']), tf.float32))

        loss = 0.0
        if n_pos > 0:
            link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=link_logits, labels=link_labels)

            def get_loss(label):
                mask = tf.equal(link_labels, label)
                weights = link_weights * tf.cast(mask, tf.float32)
                n_links = tf.reduce_sum(input_tensor=weights)
                loss = tf.reduce_sum(input_tensor=link_loss * weights)
                loss = tf.math.divide_no_nan(loss, n_links)

                return loss

            neg_link_loss = get_loss(self.conf['background_label'])
            pos_link_loss = get_loss(self.conf['text_label'])
            loss = pos_link_loss + neg_link_loss

        loss = loss / self.conf['num_replicas']

        return loss
