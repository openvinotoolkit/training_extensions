"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import torch
from torch import nn


def accuracy(predictions, targets):
    return (targets.int() == predictions.int()).float().mean()


def precision(predictions, targets):
    mask = predictions.int() == 1
    return (targets[mask].int() == 1).sum().float() / mask.sum().float()


def recall(predictions, targets):
    mask = targets.int() == 1
    return (predictions[mask].int() == 1).sum().float() / mask.sum().float()


def rpn_loss_cls(cls_targets, cls_scores, reduction='valid_mean'):
    batch_size = len(cls_targets)
    assert batch_size == cls_scores.shape[0]
    assert reduction in ('sum', 'mean', 'valid_mean')

    loss_per_image = [None for _ in range(batch_size)]
    accuracy_per_image = []
    precision_per_image = []
    recall_per_image = []
    non_negatives_num = 0.0
    for idx in range(batch_size):
        batch_cls_targets = cls_targets[idx]
        with torch.no_grad():
            weight = (batch_cls_targets >= 0).float()
            non_negatives_num += weight.sum().item()
        batch_cls_scores = cls_scores[idx].permute(1, 2, 0).reshape(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(batch_cls_scores, batch_cls_targets.float(), weight,
                                                              reduction='mean' if reduction == 'mean' else 'sum')
        valid_mask = batch_cls_targets >= 0
        pred = (batch_cls_scores > 0.5)[valid_mask].int()
        gt = batch_cls_targets[valid_mask]
        im_accuracy = accuracy(pred, gt)
        accuracy_per_image.append(im_accuracy)
        im_precision = precision(pred, gt)
        precision_per_image.append(im_precision)
        im_recall = recall(pred, gt)
        recall_per_image.append(im_recall)
        loss_per_image[idx] = loss

    loss = torch.sum(torch.stack(loss_per_image, dim=0))
    if reduction == 'valid_mean':
        non_negatives_num = max(non_negatives_num, 1.0)
        loss /= non_negatives_num
    return loss, \
           torch.mean(torch.stack(accuracy_per_image, dim=0)), \
           torch.mean(torch.stack(precision_per_image, dim=0)), \
           torch.mean(torch.stack(recall_per_image, dim=0))


def rpn_loss_reg(cls_targets, reg_targets, box_deltas, reduction='valid_mean'):
    batch_size = len(cls_targets)
    assert batch_size == len(reg_targets)
    assert batch_size == box_deltas.shape[0]
    assert reduction in ('sum', 'valid_mean')

    loss_per_image = [None for _ in range(batch_size)]
    non_negatives_num = 0.0
    for idx in range(batch_size):
        batch_rpn_bbox_deltas = box_deltas[idx, ...].permute(1, 2, 0).reshape(-1, 4)
        with torch.no_grad():
            batch_cls_targets = cls_targets[idx]
            positive_samples_mask = batch_cls_targets > 0
            non_negatives_num += (batch_cls_targets >= 0).sum().item()
        loss = smooth_l1_loss(batch_rpn_bbox_deltas[positive_samples_mask],
                              reg_targets[idx][positive_samples_mask], None, None,
                              beta=1 / 9, normalize=False)
        loss_per_image[idx] = loss
    loss = torch.sum(torch.stack(loss_per_image, dim=0))
    if reduction == 'valid_mean':
        non_negatives_num = max(non_negatives_num, 1.0)
        loss /= non_negatives_num
    return loss


def detection_loss_cls(box_class_scores, classification_targets):
    all_targets = torch.cat(classification_targets)
    assert all_targets.shape[0] == box_class_scores.shape[0]
    loss = nn.functional.cross_entropy(box_class_scores, all_targets, ignore_index=-1)
    return loss


def detection_loss_reg(box_deltas, classification_targets, regression_targets, class_agnostic_regression=False):
    device = box_deltas.device

    all_cls_targets = torch.cat(classification_targets)
    all_reg_targets = torch.cat(regression_targets, dim=0)
    assert all_cls_targets.shape[0] == all_reg_targets.shape[0]
    assert all_cls_targets.shape[0] == box_deltas.shape[0]

    # Regression loss is computed only for positive boxes.
    valid_boxes = all_cls_targets > 0
    valid_cls_targets = all_cls_targets[valid_boxes]
    valid_reg_targets = all_reg_targets[valid_boxes, :]
    deltas = box_deltas[valid_boxes, :]

    if deltas.numel() == 0:
        return torch.tensor(0, dtype=torch.float32, device=device, requires_grad=False)
    if class_agnostic_regression:
        loss = smooth_l1_loss(deltas, valid_reg_targets, None, None,
                              beta=1 / 9, normalize=True)
    else:
        with torch.no_grad():
            box_coordinates_num = 4
            boxes_num = deltas.shape[0]
            classes_num = deltas.shape[1] // box_coordinates_num
            mask = torch.zeros((boxes_num, classes_num), dtype=torch.bool, device=device)
            idx = torch.stack((torch.arange(boxes_num, device=device), valid_cls_targets), dim=1).t_()
            mask.index_put_(tuple(idx), torch.tensor([1], dtype=torch.bool, device=device))
            # Mask that selects target regression values (4 values corresponding to a box of a ground truth class)
            # from the whole output blob.
            expanded_mask = mask.unsqueeze(-1).expand(boxes_num, classes_num, box_coordinates_num).reshape(
                boxes_num, -1)
        loss = smooth_l1_loss(torch.masked_select(deltas, expanded_mask).reshape(boxes_num, box_coordinates_num),
                              valid_reg_targets, None, None, beta=1, normalize=False)
        loss /= all_reg_targets.shape[0]

    return loss


def mask_loss(mask_predictions, classification_targets, mask_targets):
    gt_labels = classification_targets
    if isinstance(gt_labels, (list, tuple)):
        gt_labels = torch.cat(gt_labels)
    gt_masks = mask_targets
    if isinstance(gt_masks, (list, tuple)):
        gt_masks = torch.cat(gt_masks, dim=0)
    rois_num = mask_predictions.shape[0]
    masks = mask_predictions[torch.arange(rois_num, device=mask_predictions.device), gt_labels]
    weight = (gt_masks > -1).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(masks.view(-1), gt_masks.view(-1),
                                                                weight.view(-1), reduction='sum')
    loss /= weight.sum()
    return loss


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights=None, bbox_outside_weights=None,
                   beta=1.0, normalize=True):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    1 / n * sum_i alpha_out[i] * SmoothL1(alpha_in[i] * (y_hat[i] - y[i])).
    n is the number of batch elements in the input predictions
    """
    box_diff = bbox_pred - bbox_targets
    if bbox_inside_weights is not None:
        box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(box_diff)
    smooth_l1_sign = (abs_in_box_diff < beta).detach().float()
    loss = smooth_l1_sign * 0.5 * torch.pow(box_diff, 2) / beta + \
           (1 - smooth_l1_sign) * (abs_in_box_diff - (0.5 * beta))
    if bbox_outside_weights is not None:
        loss = bbox_outside_weights * loss
    n = 1
    if normalize:
        n = loss.size(0)  # batch size
    loss_box = loss.view(-1).sum(0) / n
    return loss_box
