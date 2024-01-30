# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom ATSS head for OTX template."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.models.utils.misc import multi_apply
from mmdet.registry import MODELS
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
from mmdet.utils.dist_utils import reduce_mean
from torch import Tensor

from otx.algo.detection.heads.class_incremental_mixin import (
    ClassIncrementalMixin,
)
from otx.algo.detection.losses.cross_focal_loss import (
    CrossSigmoidFocalLoss,
)

if TYPE_CHECKING:
    from mmdet.utils import InstanceList, OptInstanceList


EPS = 1e-12


@MODELS.register_module()
class CustomATSSHead(ClassIncrementalMixin, ATSSHead):
    """CustomATSSHead for OTX template."""

    def __init__(
        self,
        *args,
        bg_loss_weight: float = -1.0,
        use_qfl: bool = False,
        qfl_cfg: dict | None = None,
        **kwargs,
    ):
        if use_qfl:
            kwargs["loss_cls"] = (
                qfl_cfg
                if qfl_cfg
                else {
                    "type": "QualityFocalLoss",
                    "use_sigmoid": True,
                    "beta": 2.0,
                    "loss_weight": 1.0,
                }
            )
        super().__init__(*args, **kwargs)
        self.bg_loss_weight = bg_loss_weight
        self.use_qfl = use_qfl

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        centernesses: list[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: InstanceList,
        batch_gt_instances_ignore: InstanceList | None = None,
    ) -> dict[str, Tensor]:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        if len(featmap_sizes) != self.prior_generator.num_levels:
            msg = "featmap_sizes and self.prior_generator.num_levels have different levels."
            raise ValueError(msg)

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )

        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor,
            valid_label_mask,
        ) = cls_reg_targets
        avg_factor = reduce_mean(torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = multi_apply(
            self.loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            centernesses,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            valid_label_mask,
            avg_factor=avg_factor,
        )

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = [loss_bbox / bbox_avg_factor for loss_bbox in losses_bbox]
        return {"loss_cls": losses_cls, "loss_bbox": losses_bbox, "loss_centerness": loss_centerness}

    def loss_by_feat_single(
        self,
        anchors: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        centerness: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        valid_label_mask: Tensor,
        avg_factor: float,
    ) -> tuple:
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centerness(Tensor): Centerness scores for each scale level.
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            avg_factor (float): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.
            valid_label_mask (Tensor): Label mask for consideration of ignored
                label with shape (N, num_total_anchors, 1).

        Returns:
            tuple[Tensor]: A tuple of loss components.
        """
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        valid_label_mask = valid_label_mask.reshape(-1, self.cls_out_channels)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = self._get_pos_inds(labels)

        if self.use_qfl:
            quality = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(pos_anchors, pos_bbox_targets)
            if self.reg_decoded_bbox:
                pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)

            if self.use_qfl:
                quality[pos_inds] = bbox_overlaps(pos_bbox_pred.detach(), pos_bbox_targets, is_aligned=True).clamp(
                    min=1e-6,
                )

            # regression loss
            loss_bbox = self._get_loss_bbox(pos_bbox_targets, pos_bbox_pred, centerness_targets)

            # centerness loss
            loss_centerness = self._get_loss_centerness(avg_factor, pos_centerness, centerness_targets)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.0)

        # Re-weigting BG loss
        if self.bg_loss_weight >= 0.0:
            neg_indices = labels == self.num_classes
            label_weights[neg_indices] = self.bg_loss_weight

        if self.use_qfl:
            labels = (labels, quality)  # For quality focal loss arg spec

        # classification loss
        loss_cls = self._get_loss_cls(cls_score, labels, label_weights, valid_label_mask, avg_factor)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def _get_pos_inds(self, labels: Tensor) -> Tensor:
        bg_class_ind = self.num_classes
        return ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

    def _get_loss_cls(
        self,
        cls_score: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        valid_label_mask: Tensor,
        avg_factor: Tensor,
    ) -> Tensor:
        if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                valid_label_mask=valid_label_mask,
            )
        else:
            loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor)
        return loss_cls

    def _get_loss_centerness(
        self,
        avg_factor: Tensor,
        pos_centerness: Tensor,
        centerness_targets: Tensor,
    ) -> Tensor:
        return self.loss_centerness(pos_centerness, centerness_targets, avg_factor=avg_factor)

    def _get_loss_bbox(
        self,
        pos_bbox_targets: Tensor,
        pos_bbox_pred: Tensor,
        centerness_targets: Tensor,
    ) -> Tensor:
        return self.loss_bbox(pos_bbox_pred, pos_bbox_targets, weight=centerness_targets, avg_factor=1.0)

    def get_targets(
        self,
        anchor_list: list[list[Tensor]],
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Get targets for Detection head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        However, if the detector's head loss uses CrossSigmoidFocalLoss,
        the labels_weights_list consists of (binarized label schema * weights) of batch images
        """
        return self.get_atss_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs,
        )
