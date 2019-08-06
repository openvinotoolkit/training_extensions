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


def iou(box_a, box_b):
    """Calculates IoU metric between two specified boxes.

    :param box_a: Coordinates of first box
    :param box_b: Coordinates of second box
    :return: Scalar value of UoU metric
    """

    intersect_ymin = np.maximum(box_a[0], box_b[0])
    intersect_xmin = np.maximum(box_a[1], box_b[1])
    intersect_ymax = np.minimum(box_a[2], box_b[2])
    intersect_xmax = np.minimum(box_a[3], box_b[3])

    intersect_height = np.maximum(0.0, intersect_ymax - intersect_ymin)
    intersect_width = np.maximum(0.0, intersect_xmax - intersect_xmin)

    intersect_area = intersect_height * intersect_width
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = area_a + area_b - intersect_area

    overlap_ratio = intersect_area / union_area if union_area > 0.0 else 0.0

    return overlap_ratio


def matrix_iou(set_a, set_b):
    """Calculates IoU metric between all pairs of presented sets of boxes.

    :param set_a: First set of boxes
    :param set_b: Second set of boxes
    :return: Matrix of IoU metrics
    """

    intersect_ymin = np.maximum(set_a[:, 0].reshape([-1, 1]), set_b[:, 0].reshape([1, -1]))
    intersect_xmin = np.maximum(set_a[:, 1].reshape([-1, 1]), set_b[:, 1].reshape([1, -1]))
    intersect_ymax = np.minimum(set_a[:, 2].reshape([-1, 1]), set_b[:, 2].reshape([1, -1]))
    intersect_xmax = np.minimum(set_a[:, 3].reshape([-1, 1]), set_b[:, 3].reshape([1, -1]))

    intersect_heights = np.maximum(0.0, intersect_ymax - intersect_ymin)
    intersect_widths = np.maximum(0.0, intersect_xmax - intersect_xmin)

    intersect_areas = intersect_heights * intersect_widths
    areas_set_a = ((set_a[:, 2] - set_a[:, 0]) * (set_a[:, 3] - set_a[:, 1])).reshape([-1, 1])
    areas_set_b = ((set_b[:, 2] - set_b[:, 0]) * (set_b[:, 3] - set_b[:, 1])).reshape([1, -1])

    areas_set_a[np.less(areas_set_a, 0.0)] = 0.0
    areas_set_b[np.less(areas_set_b, 0.0)] = 0.0

    union_areas = areas_set_a + areas_set_b - intersect_areas

    overlaps = intersect_areas / union_areas
    overlaps[np.less_equal(union_areas, 0.0)] = 0.0

    return overlaps
