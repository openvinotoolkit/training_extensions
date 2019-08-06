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


def iou_similarity(data_a, data_b):
    """Computes IoU metric between two sets of boxes in matrix form.

    :param data_a: First set of data of shape [M, 4]
    :param data_b: Second set of data of shape [N, 4]
    :return: IoU values of shape [M, N]
    """

    def _split_and_reshape(data, out_shape):
        """Splits bounding box coordinates into vectors of coordinates.

        :param data: Input bbox data of shape [N, 4]
        :param out_shape: Shape of each vector
        :return: List of vectors
        """

        return [tf.reshape(data[:, 0], out_shape),
                tf.reshape(data[:, 1], out_shape),
                tf.reshape(data[:, 2], out_shape),
                tf.reshape(data[:, 3], out_shape)]

    anchor_bboxes = _split_and_reshape(data_a, [-1, 1])
    ref_bboxes = _split_and_reshape(data_b, [1, -1])

    intersect_ymin = tf.maximum(anchor_bboxes[0], ref_bboxes[0])
    intersect_xmin = tf.maximum(anchor_bboxes[1], ref_bboxes[1])
    intersect_ymax = tf.minimum(anchor_bboxes[2], ref_bboxes[2])
    intersect_xmax = tf.minimum(anchor_bboxes[3], ref_bboxes[3])

    intersect_height = tf.maximum(0.0, intersect_ymax - intersect_ymin)
    intersect_width = tf.maximum(0.0, intersect_xmax - intersect_xmin)
    intersect_areas = intersect_width * intersect_height

    areas1 = (anchor_bboxes[3] - anchor_bboxes[1]) * (anchor_bboxes[2] - anchor_bboxes[0])
    areas2 = (ref_bboxes[3] - ref_bboxes[1]) * (ref_bboxes[2] - ref_bboxes[0])

    union_areas = areas1 + areas2 - intersect_areas

    out_values = tf.where(tf.greater(union_areas, 0.0), intersect_areas / union_areas, tf.zeros_like(intersect_areas))

    return out_values
