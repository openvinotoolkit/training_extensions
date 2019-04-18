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

import numpy as np
import torch
import torch.nn as nn

from ..utils.boxes import jaccard, bbox_transform_inv


def unmap(values, indices, n, dim=0, fill=-1):
    shape = list(values.shape)
    shape[dim] = n
    out = torch.full(shape, fill, dtype=values.dtype, device=values.device)
    out.index_copy_(dim, indices, values)
    return out


class RPNGTMatcher(nn.Module):
    def __init__(self, straddle_threshold=0, positive_threshold=0.5, negative_threshold=0.3,
                 positive_fraction=0.3, batch_size=512):
        super().__init__()
        self.straddle_thresh = straddle_threshold
        self.positive_overlap_range = (positive_threshold, 1.1)
        self.negative_overlap_range = (0.0, negative_threshold)
        self.ensure_closest_box = True
        self.fg_fraction = positive_fraction
        self.batch_size = batch_size

    def forward(self, boxes, gt_boxes, gt_labels, gt_ignore_labels, im_info):
        batch_size = len(gt_boxes)
        assert batch_size == len(gt_labels)
        assert batch_size == im_info.shape[0]

        rpn_cls_targets = []
        rpn_reg_targets = []
        for idx in range(batch_size):
            cls_targets, reg_targets, bbox_inw, bbox_outw = \
                self.forward_single_image(boxes.view(-1, 4), gt_boxes[idx], gt_labels[idx],
                                          gt_ignore_labels[idx], im_info[idx])
            del bbox_inw
            del bbox_outw
            rpn_cls_targets.append(cls_targets)
            rpn_reg_targets.append(reg_targets)

        return rpn_cls_targets, rpn_reg_targets

    def forward_single_image(self, boxes, gt_boxes, gt_labels, gt_ignore_labels, im_info):
        device = boxes.device
        im_height, im_width = im_info[:2]
        total_boxes_num = boxes.shape[0]

        # Filter out boxes that go outside the image more than allowed.
        # Set threshold to -1 (or a large value) to keep all boxes.
        if self.straddle_thresh >= 0:
            inds_inside = (
                (boxes[:, 0] >= -self.straddle_thresh) &
                (boxes[:, 1] >= -self.straddle_thresh) &
                (boxes[:, 2] < im_width + self.straddle_thresh + 1) &
                (boxes[:, 3] < im_height + self.straddle_thresh + 1)
            ).nonzero().view(-1)
            anchors = boxes[inds_inside, :]
        else:
            inds_inside = torch.arange(total_boxes_num, type=torch.long, device=boxes.device)
            anchors = boxes
        num_inside = len(inds_inside)

        if num_inside == 0:
            return torch.full((total_boxes_num, ), -1, dtype=torch.float32, device=device), \
                   torch.full((total_boxes_num, 4), 0, dtype=torch.float32, device=device), \
                   torch.full((total_boxes_num, 4), 0, dtype=torch.float32, device=device), \
                   torch.full((total_boxes_num, 4), 0, dtype=torch.float32, device=device)

        # Compute box labels:
        # label == 1 is positive, 0 is negative, -1 is don't care (ignore).
        labels = torch.full((num_inside, ), -1, dtype=torch.int, device=device)

        # Exclude ignored (crowd) boxes.
        gt_boxes = gt_boxes[gt_ignore_labels == 0]
        if len(gt_boxes) > 0:
            # Compute overlaps between the boxes and the GT boxes.
            anchor_by_gt_overlap = jaccard(anchors, gt_boxes)
            # For each box, amount of overlap with most overlapping GT box
            # and mapping from box to GT box that has highest overlap.
            anchor_to_gt_max, anchor_to_gt_idx = anchor_by_gt_overlap.max(dim=1)
            # For each GT box, amount of overlap with most overlapping box
            # and mapping from GT box to a box that has highest overlap.
            gt_to_anchor_max, gt_to_anchor_idx = anchor_by_gt_overlap.max(dim=0, keepdim=True)
            # Find all boxes that share the max overlap amount if it's larger than 0
            # (this includes many ties).
            gt_to_anchor_max[gt_to_anchor_max == 0.0] = 1.0
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).max(dim=1)[0]
            # Positive (foreground) label: for each GT use boxes with highest overlap (including ties).
            labels[anchors_with_max_overlap] = 1
            # Positive (foreground) label: boxes that have overlap with GT more than the threshold.
            labels[anchor_to_gt_max >= self.positive_overlap_range[0]] = 1

        # Subsample positive labels if we have too many.
        positive_mask = labels > 0
        fg_num = int(positive_mask.sum().item())
        # assert fg_num > 0
        target_fg_num = int(self.fg_fraction * self.batch_size)
        # assert target_fg_num > 0
        if target_fg_num < fg_num:
            positive_indices = positive_mask.nonzero().reshape(-1)
            # torch.randperm tends to throw a SEGFAULT in a multi-GPU setup,
            # so using numpy.random.permutation here as a workaround.
            shuffled_order = torch.from_numpy(np.random.permutation(positive_indices.numel())).to(device=device)
            # shuffled_order = torch.randperm(positive_indices.numel(), device=device)
            assert 0 < fg_num - target_fg_num <= len(shuffled_order)
            labels.index_fill_(0, positive_indices[shuffled_order[:fg_num - target_fg_num]], -1)
        else:
            target_fg_num = fg_num
        assert target_fg_num == (labels > 0).sum().item()

        # Subsample negative labels if we have too many.
        # Samples with replacement, but since the set of background indices is large,
        # most samples will not have repeats.
        target_bg_num = int((self.batch_size - (labels > 0).sum()).item())
        negative_mask = anchor_to_gt_max < self.negative_overlap_range[1]
        if negative_mask.sum().item() > target_bg_num:
            enable_inds = torch.multinomial(negative_mask.to(torch.float32), target_bg_num, replacement=True)
            labels.index_fill_(0, enable_inds, 0)

        # Get target for bounding box regression.
        bbox_targets = torch.zeros((num_inside, 4), dtype=torch.float32, device=device)
        positive_mask = labels > 0
        if positive_mask.sum() > 0:
            bbox_targets.masked_scatter_(positive_mask.unsqueeze(1), bbox_transform_inv(
                anchors[positive_mask, :], gt_boxes[anchor_to_gt_idx[positive_mask], :], (1.0, 1.0, 1.0, 1.0)
            ))

        # Bbox regression loss has the form:
        #   loss(x) = weight_outside * L(weight_inside * x)
        # Inside weights allow us to set zero loss on an element-wise basis
        # Bbox regression is only trained on positive examples so we set their
        # weights to 1 and 0 otherwise.
        positive_mask.unsqueeze_(1)
        bbox_inside_weights = torch.zeros((num_inside, 4), dtype=torch.float32, device=device)
        bbox_inside_weights.masked_fill_(positive_mask, 1.0)

        # We need to average regression loss by the total number of boxes selected.
        # Outside weights are used to scale each element-wise loss so the final
        # average over the mini-batch is correct.
        bbox_outside_weights = torch.zeros((num_inside, 4), dtype=torch.float32, device=device)
        non_negative_mask = labels >= 0
        non_negative_mask.unsqueeze_(1)
        num_examples = max(non_negative_mask.sum().item(), 1.0)
        bbox_outside_weights.masked_fill_(non_negative_mask, 1.0 / num_examples)

        # Map up to original set (i.e. all) of boxes.
        labels = unmap(labels, inds_inside, total_boxes_num, fill=-1)
        bbox_targets = unmap(bbox_targets, inds_inside, total_boxes_num, fill=0)
        bbox_inside_weights = unmap(bbox_inside_weights, inds_inside, total_boxes_num, fill=0)
        bbox_outside_weights = unmap(bbox_outside_weights, inds_inside, total_boxes_num, fill=0)

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
