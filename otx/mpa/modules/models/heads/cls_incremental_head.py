# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch

from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads.linear_head import LinearClsHead
import torch.nn.functional as F

from otx.mpa.utils.logger import get_logger

logger = get_logger()


@HEADS.register_module()
class ClsIncrHead(LinearClsHead):
    def __init__(self,
                 num_old_classes=None,
                 distillation_loss=dict(type='LwfLoss', T=2.0, loss_weight=1.0),
                 ranking_loss=None,
                 **kwargs):
        if kwargs['in_channels'] <= 0:
            raise ValueError(
                f"in_channels={kwargs['in_channels']} must be a positive integer")
        if 'hid_channels' in kwargs.keys():
            # it's for the non linear classifier header but not needed for linear head. pop out it
            logger.info("ClsIncrHead does not require 'hid_channels' configuration. remove it")
            _ = kwargs.pop('hid_channels')

        super(ClsIncrHead, self).__init__(**kwargs)
        self.num_old_classes = num_old_classes
        if num_old_classes is not None:
            if self.num_old_classes <= 0:
                raise ValueError('at least one class must be exist @ num_old_classes.')
            if not isinstance(self.num_old_classes, int):
                raise TypeError('num_old_classes must be integer type.')
        if distillation_loss is not None:
            self.compute_dist_loss = build_loss(distillation_loss)
        if ranking_loss is not None:
            self.compute_ranking_loss = build_loss(ranking_loss)
        else:
            self.compute_ranking_loss = None
        self.compute_center_loss = torch.nn.CosineSimilarity()

    def extract_prob(self, img):
        "soft label for Distillation loss"
        cls_score = self.fc(img)
        pred = cls_score.detach().cpu().numpy()[:, 0:self.num_old_classes]
        return pred

    def forward_train(self, x, gt_label, soft_label, center):
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, soft_label)
        if center is not None:
            x_old = x[gt_label < self.num_old_classes]
            center_old = center[gt_label < self.num_old_classes]
            center_loss = 0.33 * torch.mean(1 - self.compute_center_loss(x_old, center_old))
            losses['center_loss'] = center_loss
        if self.compute_ranking_loss is not None:
            ranking_loss = self.compute_ranking_loss(x, gt_label)
            losses['ranking_loss'] = ranking_loss
        return losses

    def loss(self, cls_score, gt_label, soft_label):
        losses = dict()
        num_samples = len(cls_score)

        # compute accuracy
        loss = 1.0 * self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        acc = self.compute_accuracy(cls_score, gt_label)
        if soft_label is not None:
            if not torch.is_tensor(soft_label):
                soft_label = torch.from_numpy(soft_label)
                if torch.cuda.is_available():
                    soft_label = soft_label.cuda()
            soft_label = soft_label[gt_label < self.num_old_classes]
            cls_score = cls_score[gt_label < self.num_old_classes]
            # compute loss
            losses['dist_loss'] = 0.33 * self.compute_dist_loss(cls_score[:, 0:self.num_old_classes],
                                                                soft_label[:, 0:self.num_old_classes],
                                                                avg_factor=num_samples)

        assert len(acc) == len(self.topk)
        losses['cls_loss'] = loss
        losses['accuracy'] = ({f'top-{k}': a for k, a in zip(self.topk, acc)})

        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img)

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
