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
import torch.nn.functional as F
from torch import nn

from .losses import smooth_l1_loss


class RPN(nn.Module):
    def __init__(self, dim_in, dim_internal, priors_num, class_activation_mode='softmax'):
        super().__init__()
        assert class_activation_mode in ('softmax', 'sigmoid')
        self.dim_in = dim_in
        self.dim_internal = dim_internal
        self.priors_num = priors_num
        self.dim_score = priors_num * 2 if class_activation_mode == 'softmax' else priors_num
        self.class_activation_mode = class_activation_mode

        self.conv = nn.Conv2d(dim_in, dim_internal, 3, 1, 1)
        self.cls_score = nn.Conv2d(dim_internal, self.dim_score, 1, 1, 0)
        self.bbox_deltas = nn.Conv2d(dim_internal, 4 * priors_num, 1, 1, 0)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_deltas.weight, std=0.01)
        nn.init.constant_(self.bbox_deltas.bias, 0)

    def forward(self, x):
        conv = F.relu(self.conv(x), inplace=True)
        cls_scores = self.cls_score(conv)
        bbox_deltas = self.bbox_deltas(conv)
        if self.class_activation_mode == 'softmax':
            b, c, h, w = cls_scores.shape
            cls_scores = cls_scores.view(b, 2, -1, h, w)
            cls_probs = F.softmax(cls_scores, dim=1)[:, 1].squeeze(dim=1)
        else:
            cls_probs = torch.sigmoid(cls_scores)

        return bbox_deltas, cls_scores, cls_probs


def rpn_loss(rpn_gt_matcher, im_info, gt_boxes, priors, rpn_bbox_deltas, rpn_cls_scores):
    batch_size = len(gt_boxes)
    assert batch_size == rpn_bbox_deltas.shape[0]
    assert batch_size == rpn_cls_scores.shape[0]

    device = rpn_bbox_deltas.device
    im_info = im_info.to(device)

    rpn_loss_cls = []
    rpn_loss_bbox = []
    # For each element of the batch.
    for idx in range(batch_size):
        with torch.no_grad():
            labels, bbox_targets, bbox_inw, bbox_outw = \
                rpn_gt_matcher(priors.view(-1, 4), gt_boxes[idx].to(device), None, im_info[idx, ...])

        # Classification loss.
        weight = (labels >= 0).float()
        batch_rpn_cls_scores = rpn_cls_scores[idx, 0, ...].permute(1, 2, 0).reshape(-1)
        loss_rpn_cls = nn.functional.binary_cross_entropy_with_logits(batch_rpn_cls_scores, labels.float(),
                                                                      weight, size_average=False)
        loss_rpn_cls /= weight.sum()

        # Regression loss.
        batch_rpn_bbox_deltas = rpn_bbox_deltas[idx, ...].permute(1, 2, 0).reshape(-1, 4)
        # TODO. Custom smooth L1 loss is used here. The main difference between this and out-of-the-box losses
        # is `beta` parameter, which may be not essential here. So, try to move to the standard implementation later.
        loss_rpn_bbox = smooth_l1_loss(batch_rpn_bbox_deltas, bbox_targets, bbox_inw, bbox_outw,
                                       beta=1 / 9, normalize=False)

        rpn_loss_cls.append(loss_rpn_cls)
        rpn_loss_bbox.append(loss_rpn_bbox)

    rpn_loss_cls = torch.stack(rpn_loss_cls, dim=0)
    rpn_loss_bbox = torch.stack(rpn_loss_bbox, dim=0)
    return rpn_loss_cls, rpn_loss_bbox
