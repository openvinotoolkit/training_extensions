"""
NMS Module
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from openvino.model_zoo.model_api.models.utils import nms


def multiclass_nms(
    scores: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    iou_threshold=0.45,
    max_num=200,
):
    """Multi-class NMS

    strategy: in order to perform NMS independently per class,
    we add an offset to all the boxes. The offset is dependent
    only on the class idx, and is large enough so that boxes
    from different classes do not overlap

    Args:
        scores (np.ndarray): box scores
        labels (np.ndarray): box label indices
        boxes (np.ndarray): box coordinates
        iou_threshold (float, optional): IoU threshold. Defaults to 0.45.
        max_num (int, optional): Max number of objects filter. Defaults to 200.

    Returns:
        _type_: _description_
    """
    max_coordinate = boxes.max()
    offsets = labels.astype(boxes.dtype) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(*boxes_for_nms.T, scores, iou_threshold)
    if max_num > 0:
        keep = keep[:max_num]
    keep = np.array(keep)
    return keep
