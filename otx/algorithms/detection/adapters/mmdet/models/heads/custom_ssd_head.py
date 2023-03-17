"""Custom SSD head for OTX template."""
# Copyright (C) 2018-2021 OpenMMLab
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.cnn import build_activation_layer
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.ssd_head import SSDHead
from mmdet.models.losses import smooth_l1_loss
from torch import nn

# pylint: disable=too-many-arguments, too-many-locals


@HEADS.register_module()
class CustomSSDHead(SSDHead):
    """CustomSSDHead class for OTX."""

    def __init__(self, *args, bg_loss_weight=-1.0, loss_cls=None, loss_balancing=False, **kwargs):

        super().__init__(*args, **kwargs)
        if loss_cls is None:
            loss_cls = dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                reduction="none",
                loss_weight=1.0,
            )
        self.loss_cls = build_loss(loss_cls)
        self.bg_loss_weight = bg_loss_weight
        self.loss_balancing = loss_balancing
        if self.loss_balancing:
            self.loss_weights = torch.nn.Parameter(torch.FloatTensor(2))
            for i in range(2):
                self.loss_weights.data[i] = 0.0

    # TODO: remove this internal method
    # _init_layers of CustomSSDHead(this) and of SSDHead(parent)
    # Initialize almost the same model structure.
    # However, there are subtle differences
    # Theses differences make `load_state_dict_pre_hook()` go wrong
    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        act_cfg = self.act_cfg.copy()
        act_cfg.setdefault("inplace", True)
        for in_channel, num_base_priors in zip(self.in_channels, self.num_base_priors):
            if self.use_depthwise:
                activation_layer = build_activation_layer(act_cfg)

                self.reg_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                        nn.BatchNorm2d(in_channel),
                        activation_layer,
                        nn.Conv2d(in_channel, num_base_priors * 4, kernel_size=1, padding=0),
                    )
                )
                self.cls_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                        nn.BatchNorm2d(in_channel),
                        activation_layer,
                        nn.Conv2d(in_channel, num_base_priors * self.cls_out_channels, kernel_size=1, padding=0),
                    )
                )
            else:
                self.reg_convs.append(nn.Conv2d(in_channel, num_base_priors * 4, kernel_size=3, padding=1))
                self.cls_convs.append(
                    nn.Conv2d(in_channel, num_base_priors * self.cls_out_channels, kernel_size=3, padding=1)
                )

    def loss_single(
        self,
        cls_score,
        bbox_pred,
        anchor,
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        num_total_samples,
    ):
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
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # Re-weigting BG loss
        label_weights = label_weights.reshape(-1)
        if self.bg_loss_weight >= 0.0:
            neg_indices = labels == self.num_classes
            label_weights = label_weights.clone()
            label_weights[neg_indices] = self.bg_loss_weight

        loss_cls_all = self.loss_cls(cls_score, labels, label_weights)
        if len(loss_cls_all.shape) > 1:
            loss_cls_all = loss_cls_all.sum(-1)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)

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

        # TODO: We need to verify that this is working properly.
        # pylint: disable=redundant-keyword-arg
        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples,
        )
        return loss_cls[None], loss_bbox

    def loss(self, *args, **kwargs):
        """Loss function."""
        losses = super().loss(*args, **kwargs)
        losses_cls = losses["loss_cls"]
        losses_bbox = losses["loss_bbox"]

        if self.loss_balancing:
            losses_cls, losses_bbox = self._balance_losses(losses_cls, losses_bbox)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _balance_losses(self, losses_cls, losses_reg):
        loss_cls = sum(_loss.mean() for _loss in losses_cls)
        loss_cls = torch.exp(-self.loss_weights[0]) * loss_cls + 0.5 * self.loss_weights[0]

        loss_reg = sum(_loss.mean() for _loss in losses_reg)
        loss_reg = torch.exp(-self.loss_weights[1]) * loss_reg + 0.5 * self.loss_weights[1]

        return (loss_cls, loss_reg)
