# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import torch


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy + 1), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0] + 1) *
              (box_a[:, 3] - box_a[:, 1] + 1)).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0] + 1) *
              (box_b[:, 3] - box_b[:, 1] + 1)).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def clip_boxes_to_image(boxes, height, width):
    """Clip an array of boxes to an image with the given height and width."""
    if boxes.numel() > 0:
        boxes[:, 0].clamp_(min=0, max=width - 1)
        boxes[:, 1].clamp_(min=0, max=height - 1)
        boxes[:, 2].clamp_(min=0, max=width - 1)
        boxes[:, 3].clamp_(min=0, max=height - 1)
    return boxes


def bbox_transform(boxes, deltas, bbox_xform_clip=np.log(1000. / 16.), weights=(1.0, 1.0, 1.0, 1.0)):
    device_id = boxes.device
    if boxes.shape[0] == 0:
        return torch.zeros((0, deltas.shape[1]), dtype=torch.float32, device=device_id)

    n = deltas.shape[0]
    deltas = deltas.view(n, -1, 4).permute(2, 0, 1)
    boxes = boxes.permute(1, 0).view(4, -1, 1)
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32, device=device_id).view(4, 1, 1)
        deltas /= weights

    dxy = deltas[:2]
    dwh = deltas[2:]
    dwh.clamp_(max=bbox_xform_clip).exp_().mul_(0.5)
    dxy -= dwh
    dwh *= 2
    dwh += dxy

    dx0, dy0, dx1, dy1 = deltas
    x0, y0, x1, y1 = boxes

    w = x1 - x0 + 1
    h = y1 - y0 + 1
    cx = x0 + 0.5 * w
    cy = y0 + 0.5 * h
    x0new = w * dx0 + cx
    y0new = h * dy0 + cy
    x1new = w * dx1 + cx - 1
    y1new = h * dy1 + cy - 1

    pred_boxes = torch.cat((x0new, y0new, x1new, y1new), 0).view(4, n, -1).permute(1, 2, 0).reshape(n, -1)

    return pred_boxes


def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def bbox_transform_inv(boxes, gt_boxes, weights=(10., 10., 5., 5.)):
    """
    Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """

    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = gt_widths / ex_widths
    targets_dw.log_()
    targets_dw *= ww
    targets_dh = gt_heights / ex_heights
    targets_dh.log_()
    targets_dh *= wh

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=0)
    targets.transpose_(1, 0)
    return targets
