"""Custom SSD head for OTX template."""
# Copyright (C) 2018-2021 OpenMMLab
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.cnn import build_activation_layer
from mmdet.core import anchor_inside_flags, unmap
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.ssd_head import SSDHead
from mmdet.models.losses import smooth_l1_loss
from torch import nn

from otx.algorithms.detection.adapters.mmdet.models.heads.cross_dataset_detector_head import TrackingLossDynamicsMixIn
from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import (
    TrackingLossType,
)

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
            anchor (Tensor): Box reference for each scale level with shape
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
        pos_inds = self._get_pos_inds(labels)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls = self._get_loss_cls(num_total_samples, loss_cls_all, pos_inds, topk_loss_cls_neg)

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        # TODO: We need to verify that this is working properly.
        # pylint: disable=redundant-keyword-arg
        loss_bbox = self._get_loss_bbox(bbox_pred, bbox_targets, bbox_weights, num_total_samples)
        return loss_cls[None], loss_bbox

    def _get_pos_inds(self, labels):
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        return pos_inds

    def _get_loss_bbox(self, bbox_pred, bbox_targets, bbox_weights, num_total_samples):
        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples,
        )

        return loss_bbox

    def _get_loss_cls(self, num_total_samples, loss_cls_all, pos_inds, topk_loss_cls_neg):
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        return loss_cls

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Loss function."""
        losses = super().loss(cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
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


@HEADS.register_module()
class CustomSSDHeadTrackingLossDynamics(TrackingLossDynamicsMixIn, CustomSSDHead):
    """CustomSSDHead which supports tracking loss dynamics."""

    tracking_loss_types = (TrackingLossType.cls, TrackingLossType.bbox, TrackingLossType.centerness)

    @TrackingLossDynamicsMixIn._wrap_loss
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss from the head and prepare for loss dynamics tracking."""
        return super().loss(cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)

    @TrackingLossDynamicsMixIn._wrap_loss_single
    def loss_single(
        self, cls_score, bbox_pred, anchor, labels, label_weights, bbox_targets, bbox_weights, num_total_samples
    ):
        """Compute loss of a single image and increase `self.cur_loss_idx` counter for loss dynamics tracking."""
        return super().loss_single(
            cls_score, bbox_pred, anchor, labels, label_weights, bbox_targets, bbox_weights, num_total_samples
        )

    def _get_loss_cls(self, num_total_samples, loss_cls_all, pos_inds, topk_loss_cls_neg):
        loss_cls_pos = loss_cls_all[pos_inds]
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos.sum() + loss_cls_neg) / num_total_samples

        self._store_loss_dyns(loss_cls_pos.detach(), TrackingLossType.cls)
        return loss_cls

    def _get_loss_bbox(self, bbox_pred, bbox_targets, bbox_weights, num_total_samples):
        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples,
            reduction="none",
        )

        self._store_loss_dyns(loss_bbox[self.pos_inds].detach().mean(-1), TrackingLossType.bbox)
        return self._postprocess_loss(loss_bbox, reduction="mean", avg_factor=num_total_samples)

    @TrackingLossDynamicsMixIn._wrap_get_targets(concatenate_last=True)
    def get_targets(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
        return_sampling_results=False,
    ):
        """Get targets."""
        return super().get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            img_metas,
            gt_bboxes_ignore_list,
            gt_labels_list,
            label_channels,
            unmap_outputs,
            return_sampling_results,
        )

    def _get_targets_single(
        self,
        flat_anchors,
        valid_flags,
        gt_bboxes,
        gt_bboxes_ignore,
        gt_labels,
        img_meta,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Compute regression and classification targets for anchors in a single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, img_meta["img_shape"][:2], self.train_cfg.allowed_border
        )
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        ########## What we changed from the original mmdet code ###############
        # Store all_pos_assigned_gt_inds to member variable
        # to look up training loss dynamics for each gt_bboxes afterwards
        pos_assigned_gt_inds = anchors.new_full((num_valid_anchors,), -1, dtype=torch.long)
        if len(pos_inds) > 0:
            pos_assigned_gt_inds[pos_inds] = (
                self.cur_batch_idx * self.max_gt_bboxes_len + sampling_result.pos_assigned_gt_inds
            )
        if unmap_outputs:
            pos_assigned_gt_inds = unmap(pos_assigned_gt_inds, num_total_anchors, inside_flags, fill=-1)
        self.pos_assigned_gt_inds_list += [pos_assigned_gt_inds]
        self.cur_batch_idx += 1
        ########################################################################

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result)
