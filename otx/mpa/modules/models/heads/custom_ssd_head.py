# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import smooth_l1_loss
from mmdet.models.dense_heads.ssd_head import SSDHead


@HEADS.register_module()
class CustomSSDHead(SSDHead):
    def __init__(
        self,
        *args,
        bg_loss_weight=-1.0,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='none',
            loss_weight=1.0
        ),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_cls = build_loss(loss_cls)
        self.bg_loss_weight = bg_loss_weight

    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.num_classes)).nonzero().reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero().view(-1)

        # Re-weigting BG loss
        label_weights = label_weights.reshape(-1)
        if self.bg_loss_weight >= 0.0:
            neg_indices = (labels == self.num_classes)
            label_weights = label_weights.clone()
            label_weights[neg_indices] = self.bg_loss_weight

        loss_cls_all = self.loss_cls(cls_score, labels, label_weights)
        if len(loss_cls_all.shape) > 1:
            loss_cls_all = loss_cls_all.sum(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox
