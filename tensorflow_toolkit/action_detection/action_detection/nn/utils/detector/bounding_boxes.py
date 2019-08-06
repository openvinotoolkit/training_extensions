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


def center_encode(bboxes, priors, variance):
    """Carry out encoding bounding box coordinate into SSD format.

    :param bboxes: Input bounding boxes
    :param priors: Set of prior boxes
    :param variance: List of variances
    :return: Encoded bounding box coordinates
    """

    priors_height = priors[:, 2] - priors[:, 0]
    priors_width = priors[:, 3] - priors[:, 1]
    priors_center_y = 0.5 * (priors[:, 0] + priors[:, 2])
    priors_center_x = 0.5 * (priors[:, 1] + priors[:, 3])

    bboxes_height = bboxes[:, 2] - bboxes[:, 0]
    bboxes_width = bboxes[:, 3] - bboxes[:, 1]
    bboxes_center_y = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
    bboxes_center_x = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

    encoded_bbox_ymin = (bboxes_center_y - priors_center_y) / (priors_height * variance[0])
    encoded_bbox_xmin = (bboxes_center_x - priors_center_x) / (priors_width * variance[1])
    encoded_bbox_ymax = tf.log(bboxes_height / priors_height) / variance[2]
    encoded_bbox_xmax = tf.log(bboxes_width / priors_width) / variance[3]

    return tf.stack([encoded_bbox_ymin, encoded_bbox_xmin, encoded_bbox_ymax, encoded_bbox_xmax], axis=1)


def center_decode(encoded_locs, priors, variance, clip=False):
    """Carry out decoding bounding box coordinate from SSD format.

    :param encoded_locs: Encoded bounding boxes
    :param priors: Set of prior boxes
    :param variance: List of variances
    :param clip: Whether to clip coordinates into [0, 1] interval
    :return: Decoded bounding boxes
    """

    output_shape = tf.shape(encoded_locs)
    num_priors = tf.shape(priors)[0]
    encoded_locs = tf.reshape(encoded_locs, [-1, num_priors, 4])

    priors_height = priors[:, 2] - priors[:, 0]
    priors_width = priors[:, 3] - priors[:, 1]
    priors_center_y = 0.5 * (priors[:, 0] + priors[:, 2])
    priors_center_x = 0.5 * (priors[:, 1] + priors[:, 3])

    bboxes_center_y = encoded_locs[:, :, 0] * priors_height * variance[0] + priors_center_y
    bboxes_center_x = encoded_locs[:, :, 1] * priors_width * variance[1] + priors_center_x
    bboxes_height = tf.exp(encoded_locs[:, :, 2] * variance[2]) * priors_height
    bboxes_width = tf.exp(encoded_locs[:, :, 3] * variance[3]) * priors_width

    ymin = tf.expand_dims(bboxes_center_y - 0.5 * bboxes_height, axis=2)
    xmin = tf.expand_dims(bboxes_center_x - 0.5 * bboxes_width, axis=2)
    ymax = tf.expand_dims(bboxes_center_y + 0.5 * bboxes_height, axis=2)
    xmax = tf.expand_dims(bboxes_center_x + 0.5 * bboxes_width, axis=2)

    decoded_locs = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    if clip:
        decoded_locs = tf.clip_by_value(decoded_locs, 0., 1.)

    decoded_locs = tf.reshape(decoded_locs, output_shape)

    return decoded_locs
