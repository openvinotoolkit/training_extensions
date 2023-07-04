"""Custom VFNet head for OTX template."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps, distance2bbox, reduce_mean
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.vfnet_head import VFNetHead

from otx.algorithms.detection.adapters.mmdet.models.heads.cross_dataset_detector_head import (
    CrossDatasetDetectorHead,
    TrackingLossDynamicsMixIn,
)
from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import TrackingLossType
from otx.algorithms.detection.adapters.mmdet.models.losses.cross_focal_loss import (
    CrossSigmoidFocalLoss,
)

from .custom_atss_head import CustomATSSHeadTrackingLossDynamics

# pylint: disable=too-many-ancestors, too-many-arguments, too-many-statements, too-many-locals


@HEADS.register_module()
class CustomVFNetHead(CrossDatasetDetectorHead, VFNetHead):
    """CustomVFNetHead class for OTX."""

    def __init__(self, *args, bg_loss_weight=-1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_loss_weight = bg_loss_weight

    @force_fp32(apply_to=("cls_scores", "bbox_preds", "bbox_preds_refine"))
    def loss(self, cls_scores, bbox_preds, bbox_preds_refine, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level, each is a 4D-tensor, the channel
                number is num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, label_weights, bbox_targets, _, valid_label_mask = self.get_targets(
            cls_scores, all_level_points, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore
        )

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and bbox_preds_refine
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous() for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous() for bbox_pred in bbox_preds]
        flatten_bbox_preds_refine = [
            bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4).contiguous() for bbox_pred_refine in bbox_preds_refine
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)

        valid_label_mask = [
            valid_mask.reshape(-1, self.cls_out_channels).contiguous() for valid_mask in valid_label_mask
        ]
        flatten_valid_label_mask = torch.cat(valid_label_mask)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = self._get_pos_inds(flatten_labels, bg_class_ind)
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        # sync num_pos across all gpus
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos
        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_points = flatten_points[pos_inds]

            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
            iou_targets_ini = bbox_overlaps(
                pos_decoded_bbox_preds, pos_decoded_target_preds.detach(), is_aligned=True
            ).clamp(min=1e-6)
            bbox_weights_ini = iou_targets_ini.clone().detach()
            iou_targets_ini_avg_per_gpu = reduce_mean(bbox_weights_ini.sum()).item()
            bbox_avg_factor_ini = max(iou_targets_ini_avg_per_gpu, 1.0)
            loss_bbox = self._get_loss_bbox(
                pos_decoded_bbox_preds, pos_decoded_target_preds, bbox_weights_ini, bbox_avg_factor_ini
            )

            pos_decoded_bbox_preds_refine = distance2bbox(pos_points, pos_bbox_preds_refine)
            iou_targets_rf = bbox_overlaps(
                pos_decoded_bbox_preds_refine, pos_decoded_target_preds.detach(), is_aligned=True
            ).clamp(min=1e-6)
            bbox_weights_rf = iou_targets_rf.clone().detach()
            iou_targets_rf_avg_per_gpu = reduce_mean(bbox_weights_rf.sum()).item()
            bbox_avg_factor_rf = max(iou_targets_rf_avg_per_gpu, 1.0)
            loss_bbox_refine = self._get_loss_bbox_refine(
                pos_decoded_target_preds, pos_decoded_bbox_preds_refine, bbox_weights_rf, bbox_avg_factor_rf
            )

            # build IoU-aware cls_score targets
            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
        # Re-weigting BG loss
        if self.bg_loss_weight >= 0.0:
            neg_indices = flatten_labels == self.num_classes
            label_weights[neg_indices] = self.bg_loss_weight

        loss_cls = self._get_loss_cls(
            label_weights,
            flatten_cls_scores,
            flatten_labels,
            flatten_valid_label_mask,
            num_pos_avg_per_gpu,
            cls_iou_targets,
        )

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_bbox_rf=loss_bbox_refine)

    def _get_loss_cls(
        self,
        label_weights,
        flatten_cls_scores,
        flatten_labels,
        flatten_valid_label_mask,
        num_pos_avg_per_gpu,
        cls_iou_targets,
    ):
        if self.use_vfl:
            if label_weights is not None:
                label_weights = label_weights.unsqueeze(-1)
            if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
                loss_cls = self.loss_cls(
                    flatten_cls_scores,
                    cls_iou_targets,
                    weight=label_weights,
                    avg_factor=num_pos_avg_per_gpu,
                    use_vfl=self.use_vfl,
                    valid_label_mask=flatten_valid_label_mask,
                )
            else:
                loss_cls = self.loss_cls(
                    flatten_cls_scores, cls_iou_targets, weight=label_weights, avg_factor=num_pos_avg_per_gpu
                )
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_labels, weight=label_weights, avg_factor=num_pos_avg_per_gpu
            )

        return loss_cls

    def _get_loss_bbox_refine(
        self, pos_decoded_target_preds, pos_decoded_bbox_preds_refine, bbox_weights_rf, bbox_avg_factor_rf
    ):
        return self.loss_bbox_refine(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            weight=bbox_weights_rf,
            avg_factor=bbox_avg_factor_rf,
        )

    def _get_loss_bbox(self, pos_decoded_bbox_preds, pos_decoded_target_preds, bbox_weights_ini, bbox_avg_factor_ini):
        return self.loss_bbox(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            weight=bbox_weights_ini,
            avg_factor=bbox_avg_factor_ini,
        )

    def _get_pos_inds(self, flatten_labels, bg_class_ind):
        pos_inds = torch.where(((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]
        return pos_inds

    def get_targets(self, cls_scores, mlvl_points, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore):
        """A wrapper for computing ATSS and FCOS targets for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor/None): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor/None): Bbox weights of all levels.
        """
        if self.use_atss:
            return self.vfnet_to_atss_targets(
                cls_scores, mlvl_points, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore
            )
        self.norm_on_bbox = False
        return self.get_fcos_targets(mlvl_points, gt_bboxes, gt_labels, img_metas)


@HEADS.register_module()
class CustomVFNetHeadTrackingLossDynamics(TrackingLossDynamicsMixIn, CustomVFNetHead):
    """CustomVFNetHead which supports tracking loss dynamics."""

    tracking_loss_types = (TrackingLossType.cls, TrackingLossType.bbox, TrackingLossType.bbox_refine)

    def __init__(self, *args, bg_loss_weight=-1, **kwargs):
        super().__init__(*args, bg_loss_weight=bg_loss_weight, **kwargs)

        if not self.use_atss:
            raise NotImplementedError(
                "Loss dynamics tracking for VFNetHead with use_atss=False is currently not supported."
            )

    @TrackingLossDynamicsMixIn._wrap_get_targets(concatenate_last=True, flatten=True)
    def get_atss_targets(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Extract ATSS targets and store `self.all_pos_assigned_gt_inds` for tracking loss dynamics."""
        return super().get_atss_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            img_metas,
            gt_bboxes_ignore_list,
            gt_labels_list,
            label_channels,
            unmap_outputs,
        )

    def _get_target_single(self, *args, **kwargs):
        return CustomATSSHeadTrackingLossDynamics._get_target_single(self, *args, **kwargs)

    @TrackingLossDynamicsMixIn._wrap_loss
    def loss(self, cls_scores, bbox_preds, bbox_preds_refine, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Obtain detection task losses and store loss dynamics."""
        return super().loss(
            cls_scores, bbox_preds, bbox_preds_refine, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore
        )

    def _get_loss_cls(
        self,
        label_weights,
        flatten_cls_scores,
        flatten_labels,
        flatten_valid_label_mask,
        num_pos_avg_per_gpu,
        cls_iou_targets,
    ):
        if self.use_vfl:
            if label_weights is not None:
                label_weights = label_weights.unsqueeze(-1)
            if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
                loss_cls = self.loss_cls(
                    flatten_cls_scores,
                    cls_iou_targets,
                    weight=label_weights,
                    avg_factor=num_pos_avg_per_gpu,
                    use_vfl=self.use_vfl,
                    valid_label_mask=flatten_valid_label_mask,
                    reduction_override="none",
                )
            else:
                loss_cls = self.loss_cls(
                    flatten_cls_scores,
                    cls_iou_targets,
                    weight=label_weights,
                    avg_factor=num_pos_avg_per_gpu,
                    reduction_override="none",
                )
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu,
                reduction_override="none",
            )

        if len(self.pos_inds) > 0:
            self._store_loss_dyns(loss_cls[self.pos_inds].detach().mean(-1), TrackingLossType.cls)

        return self._postprocess_loss(loss_cls, self.loss_cls.reduction, avg_factor=num_pos_avg_per_gpu)

    def _get_loss_bbox_refine(
        self, pos_decoded_target_preds, pos_decoded_bbox_preds_refine, bbox_weights_rf, bbox_avg_factor_rf
    ):
        loss_bbox_refine = self.loss_bbox_refine(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            weight=bbox_weights_rf,
            avg_factor=bbox_avg_factor_rf,
            reduction_override="none",
        )
        self._store_loss_dyns(loss_bbox_refine.detach(), TrackingLossType.bbox_refine)

        return self._postprocess_loss(loss_bbox_refine, self.loss_cls.reduction, avg_factor=bbox_avg_factor_rf)

    def _get_loss_bbox(self, pos_decoded_bbox_preds, pos_decoded_target_preds, bbox_weights_ini, bbox_avg_factor_ini):
        loss_bbox = self.loss_bbox(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            weight=bbox_weights_ini,
            avg_factor=bbox_avg_factor_ini,
            reduction_override="none",
        )
        self._store_loss_dyns(loss_bbox.detach(), TrackingLossType.bbox)

        return self._postprocess_loss(loss_bbox, self.loss_cls.reduction, avg_factor=bbox_avg_factor_ini)

    def _get_pos_inds(self, labels, *args, **kwargs):
        pos_inds = CustomVFNetHead._get_pos_inds(self, labels, *args, **kwargs)

        if len(pos_inds) > 0:
            gt_inds = self.all_pos_assigned_gt_inds[pos_inds].cpu()

            self.batch_inds = gt_inds // self.max_gt_bboxes_len
            self.bbox_inds = gt_inds % self.max_gt_bboxes_len

        self.pos_inds = pos_inds
        return pos_inds
