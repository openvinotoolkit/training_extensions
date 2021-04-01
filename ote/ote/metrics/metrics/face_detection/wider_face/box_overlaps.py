# https://github.com/wondervictor/WiderFace-Evaluation/blob/master/box_overlaps.pyx
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

""" Module for computing intersection over union. """

import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes_num = boxes.shape[0]
    queries_num = query_boxes.shape[0]
    overlaps = np.zeros((boxes_num, queries_num), dtype=np.float)
    for k in range(queries_num):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for box_id in range(boxes_num):
            int_width = (
                min(boxes[box_id, 2], query_boxes[k, 2]) -
                max(boxes[box_id, 0], query_boxes[k, 0]) + 1
            )
            if int_width > 0:
                int_height = (
                    min(boxes[box_id, 3], query_boxes[k, 3]) -
                    max(boxes[box_id, 1], query_boxes[k, 1]) + 1
                )
                if int_height > 0:
                    union_area = float(
                        (boxes[box_id, 2] - boxes[box_id, 0] + 1) *
                        (boxes[box_id, 3] - boxes[box_id, 1] + 1) +
                        box_area - int_width * int_height
                    )
                    overlaps[box_id, k] = int_width * int_height / union_area
    return overlaps
