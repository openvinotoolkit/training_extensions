"""
 Copyright (c) 2020 Intel Corporation

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
from segmentoly.utils.boxes import jaccard, bbox_transform_inv
from segmentoly.utils.segms import polys_to_mask_wrt_box


class ProposalGTMatcher(nn.Module):
    def __init__(self, positive_threshold=0.5, negative_threshold=0.3,
                 positive_fraction=0.3, ignore_threshold=0.5, batch_size=512,
                 target_mask_size=(14, 14)):
        super().__init__()
        self.positive_overlap_range = (positive_threshold, 1.1)
        self.negative_overlap_range = (0.0, negative_threshold)
        self.ignore_threshold = ignore_threshold
        self.ensure_closest_box = True
        self.fg_fraction = positive_fraction
        self.batch_size = batch_size
        self.target_mask_size = target_mask_size

    def forward(self, boxes, gt_boxes, gt_labels, gt_masks=None, gt_texts=None):
        batch_size = len(gt_boxes)
        assert batch_size == len(gt_labels)

        sampled_boxes = []
        cls_targets = []
        reg_targets = []
        mask_targets = []
        text_targets = []
        for idx in range(batch_size):
            im_gt_masks = gt_masks[idx] if gt_masks is not None else None
            im_gt_texts = gt_texts[idx] if gt_texts is not None else None
            image_sampled_boxes, image_cls_targets, image_reg_targets, image_mask_targets, image_text_targets = \
                self.forward_single_image(boxes[idx], gt_boxes[idx], gt_labels[idx], im_gt_masks,
                                          im_gt_texts)
            sampled_boxes.append(image_sampled_boxes)
            cls_targets.append(image_cls_targets)
            reg_targets.append(image_reg_targets)
            mask_targets.append(image_mask_targets)
            text_targets.append(image_text_targets)
        if gt_texts is None:
            return sampled_boxes, cls_targets, reg_targets, mask_targets
        return sampled_boxes, cls_targets, reg_targets, mask_targets, text_targets

    def forward_single_image(self, boxes, gt_boxes, gt_labels, gt_masks=None, gt_texts=None):
        device = boxes.device

        # Add ground truth boxes to also sample those as positives later.
        boxes = torch.cat((gt_boxes, boxes.view(-1, 4)), dim=0)
        boxes_num = boxes.shape[0]
        box_to_label = torch.zeros(boxes_num, dtype=torch.long, device=device)

        if len(gt_boxes) > 0:
            # Compute overlaps between the boxes and the GT boxes.
            box_by_gt_overlap = jaccard(boxes, gt_boxes)
            # For each box, amount of overlap with most overlapping GT box
            # and mapping from box to GT box that has highest overlap.
            box_to_gt_max, box_to_gt_idx = box_by_gt_overlap.max(dim=1)
            matched_boxes_mask = box_to_gt_max > 0
            # Record max overlaps with the class of the appropriate gt box
            matched_box_to_label = gt_labels[box_to_gt_idx[matched_boxes_mask]]
            box_to_label.masked_scatter_(matched_boxes_mask, matched_box_to_label.long())

        # Subsample positive labels if we have too many.
        positive_mask = box_to_gt_max >= self.positive_overlap_range[0]
        fg_num = int(positive_mask.sum().item())
        target_fg_num = int(self.fg_fraction * self.batch_size)
        # Due to issues with torch.multinomial do subsampling in a loop
        # to ensure proper number of samples to be selected.
        while target_fg_num < fg_num:
            disable_inds = torch.multinomial(positive_mask.to(torch.float32),
                                             fg_num - target_fg_num, replacement=False)
            positive_mask[disable_inds] = 0
            fg_num = int(positive_mask.sum().item())
        target_fg_num = fg_num
        assert target_fg_num == positive_mask.sum().item()

        # Subsample negative labels if we have too many.
        negative_mask = (self.negative_overlap_range[0] <= box_to_gt_max) & \
                        (box_to_gt_max < self.negative_overlap_range[1])
        bg_num = int(negative_mask.sum().item())
        assert bg_num > 0
        target_bg_num = self.batch_size - target_fg_num
        assert target_bg_num > 0
        if target_bg_num < bg_num:
            negative_indices = negative_mask.nonzero().reshape(-1)
            assert negative_indices.numel() > 0
            # torch.randperm tends to throw a SEGFAULT in a multi-GPU setup,
            # so using numpy.random.permutation here as a workaround.
            shuffled_order = torch.from_numpy(np.random.permutation(negative_indices.numel())).to(
                device=device)
            # shuffled_order = torch.randperm(negative_indices.numel(), device=device)
            assert 0 < bg_num - target_bg_num < len(shuffled_order)
            assert (shuffled_order < negative_indices.numel()).all()
            negative_mask.index_fill_(0, negative_indices[shuffled_order[:bg_num - target_bg_num]],
                                      0)
        else:
            target_bg_num = bg_num
        assert target_bg_num == negative_mask.sum().item()

        sampled_boxes = torch.cat((boxes[positive_mask, :], boxes[negative_mask, :]), dim=0)
        sampled_labels = torch.cat((box_to_label[positive_mask],
                                    torch.zeros(target_bg_num, dtype=torch.long, device=device)))

        # Get target for bounding box regression.
        sampled_gt_boxes = gt_boxes[box_to_gt_idx[positive_mask], :]
        sampled_boxes_targets = torch.zeros((target_bg_num, 4), dtype=torch.float32, device=device)
        if sampled_gt_boxes.numel() > 0:
            sampled_boxes_targets = torch.cat((bbox_transform_inv(sampled_boxes[:target_fg_num, :],
                                                                  sampled_gt_boxes),
                                               sampled_boxes_targets), dim=0)

        sampled_masks_targets = None
        if gt_masks is not None:
            sampled_boxes_cpu = sampled_boxes.cpu().detach().numpy()
            # Get target for box segmentation.
            sampled_gt_masks = [gt_masks[i] for i in box_to_gt_idx[positive_mask]]
            sampled_masks_targets = []
            for i in range(target_fg_num):
                gt_polygon = sampled_gt_masks[i]
                box = sampled_boxes_cpu[i]
                mask = polys_to_mask_wrt_box(gt_polygon, box, self.target_mask_size)
                sampled_masks_targets.append(
                    torch.tensor(mask, dtype=torch.float32).reshape(*self.target_mask_size))
            if len(sampled_masks_targets) > 0:
                sampled_masks_targets = torch.cat((torch.stack(sampled_masks_targets, dim=0),
                                                   torch.full(
                                                       (target_bg_num, *self.target_mask_size), -1,
                                                       dtype=torch.float32)),
                                                  dim=0).to(device)
            else:
                sampled_masks_targets = torch.empty((0, *self.target_mask_size),
                                                    dtype=torch.float32, device=device)

        sampled_texts_targets = None
        if gt_texts is not None:
            sampled_gt_texts = [gt_texts[i] for i in box_to_gt_idx[positive_mask]]
            sampled_gt_texts = [
                sampled_gt_texts[i] if box_to_gt_max[positive_mask][i] >= 1.0 else [] for i in
                range(len(sampled_gt_texts))]
            sampled_texts_targets = sampled_gt_texts + [[] for _ in range(target_bg_num)]

        return sampled_boxes, sampled_labels, sampled_boxes_targets, sampled_masks_targets, sampled_texts_targets
