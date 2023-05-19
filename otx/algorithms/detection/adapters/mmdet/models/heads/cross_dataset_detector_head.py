"""Cross Dataset Detector head for Ignore labels."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools
import inspect
from collections import defaultdict
from typing import Dict, Tuple

import torch
from mmdet.core import images_to_levels, multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.losses.utils import weight_reduce_loss

from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import (
    LossAccumulator,
    TrackingLossType,
)

# TODO: Need to fix pylint issues
# pylint: disable=too-many-locals, too-many-arguments, abstract-method


@HEADS.register_module()
class CrossDatasetDetectorHead(BaseDenseHead):
    """Head class for Ignore labels."""

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
        """Get targets for Detection head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        However, if the detector's head loss uses CrossSigmoidFocalLoss,
        the labels_weights_list consists of (binarized label schema * weights) of batch images
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
        )
        # no valid anchors
        if not all(labels is not None for labels in all_labels):
            return None
        # sampled anchors of all images
        num_total_pos = sum(max(inds.numel(), 1) for inds in pos_inds_list)
        num_total_neg = sum(max(inds.numel(), 1) for inds in neg_inds_list)
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        valid_label_mask = self.get_valid_label_mask(img_metas=img_metas, all_labels=all_labels)
        valid_label_mask = [i.to(anchor_list[0].device) for i in valid_label_mask]
        if len(valid_label_mask) > 0:
            valid_label_mask = images_to_levels(valid_label_mask, num_level_anchors)

        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return (
            anchors_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            valid_label_mask,
            num_total_pos,
            num_total_neg,
        )

    def get_fcos_targets(self, points, gt_bboxes_list, gt_labels_list, img_metas):
        """Compute regression, classification and centerss targets for points in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information for the image.

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
        )

        # split to per img, per level
        valid_label_mask = self.get_valid_label_mask(img_metas=img_metas, all_labels=labels_list)
        valid_label_mask = [i.to(points[0].device) for i in valid_label_mask]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        label_weights, bbox_weights = None, None
        return concat_lvl_labels, label_weights, concat_lvl_bbox_targets, bbox_weights, valid_label_mask

    def vfnet_to_atss_targets(self, cls_scores, mlvl_points, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """A wrapper for computing ATSS targets for points in multiple images.

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
                ignored, shape (num_ignored_gts, 4). Default: None.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_atss_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            unmap_outputs=True,
        )
        if cls_reg_targets is None:
            return None

        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            valid_label_mask,
            num_total_pos,  # pylint: disable=unused-variable
            num_total_neg,  # pylint: disable=unused-variable
        ) = cls_reg_targets

        bbox_targets_list = [bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list]

        num_imgs = len(img_metas)
        # transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format
        bbox_targets_list = self.transform_bbox_targets(bbox_targets_list, mlvl_points, num_imgs)

        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [label_weights.reshape(-1) for label_weights in label_weights_list]
        bbox_weights_list = [bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights, bbox_targets_list, bbox_weights, valid_label_mask

    def get_valid_label_mask(self, img_metas, all_labels, use_bg=False):
        """Getter function valid_label_mask."""
        num_classes = self.num_classes + 1 if use_bg else self.num_classes
        valid_label_mask = []
        for i, meta in enumerate(img_metas):
            mask = torch.Tensor([1 for _ in range(num_classes)])
            if "ignored_labels" in meta and len(meta["ignored_labels"]) > 0:
                mask[meta["ignored_labels"]] = 0
                if use_bg:
                    mask[self.num_classes] = 0
            mask = mask.repeat(len(all_labels[i]), 1)
            valid_label_mask.append(mask)
        return valid_label_mask


@HEADS.register_module()
class TrackingLossDynamicsMixIn:
    """Mix-In class for tracking loss dynamics."""

    tracking_loss_types: Tuple[TrackingLossType, ...] = ()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._loss_dyns: Dict[TrackingLossType, Dict[Tuple[int, int], LossAccumulator]] = {}

    def _get_pos_inds(self, labels):
        pos_inds = super()._get_pos_inds(labels)

        if len(pos_inds) > 0:
            pos_assigned_gt_inds = self.all_pos_assigned_gt_inds[self.cur_loss_idx].reshape(-1)

            gt_inds = pos_assigned_gt_inds[pos_inds].cpu()

            self.batch_inds = gt_inds // self.max_gt_bboxes_len
            self.bbox_inds = gt_inds % self.max_gt_bboxes_len

        self.pos_inds = pos_inds
        return pos_inds

    def _store_loss_dyns(self, losses: torch.Tensor, key: TrackingLossType) -> None:
        if len(self.pos_inds) == 0:
            return

        loss_dyns = self.loss_dyns[key]
        for batch_idx, bbox_idx, loss_item in zip(self.batch_inds, self.bbox_inds, losses.detach().cpu()):
            loss_dyns[(batch_idx.item(), bbox_idx.item())].add(loss_item.item())

    def _postprocess_loss(self, losses: torch.Tensor, reduction: str, avg_factor: float) -> torch.Tensor:
        return weight_reduce_loss(losses, reduction=reduction, avg_factor=avg_factor)

    @property
    def loss_dyns(self) -> Dict[TrackingLossType, Dict[Tuple[int, int], LossAccumulator]]:
        """Loss dynamics dict."""
        return self._loss_dyns

    @staticmethod
    def _wrap_loss(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self.cur_loss_idx = 0
            self._loss_dyns = {loss_type: defaultdict(LossAccumulator) for loss_type in self.tracking_loss_types}
            losses = func(self, *args, **kwargs)
            return losses

        return wrapper

    @staticmethod
    def _wrap_loss_single(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            losses = func(self, *args, **kwargs)
            self.cur_loss_idx += 1
            return losses

        return wrapper

    @staticmethod
    def _wrap_get_targets(concatenate_last: bool = False, flatten: bool = False):
        def wrapper_with_option(func):
            signature = inspect.signature(func)

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                anchor_list = gt_bboxes_list = None
                for idx, key in enumerate(signature.parameters):
                    if key == "anchor_list":
                        anchor_list = kwargs.get(key) if key in kwargs else args[idx - 1]
                    elif key == "gt_bboxes_list":
                        gt_bboxes_list = kwargs.get(key) if key in kwargs else args[idx - 1]

                assert anchor_list is not None and gt_bboxes_list is not None
                num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
                self.max_gt_bboxes_len = max([len(gt_bboxes) for gt_bboxes in gt_bboxes_list])
                self.cur_batch_idx = 0
                self.pos_assigned_gt_inds_list = []
                targets = func(self, *args, **kwargs)
                self.all_pos_assigned_gt_inds = images_to_levels(self.pos_assigned_gt_inds_list, num_level_anchors)
                if flatten:
                    self.all_pos_assigned_gt_inds = [gt_ind.reshape(-1) for gt_ind in self.all_pos_assigned_gt_inds]
                if concatenate_last:
                    self.all_pos_assigned_gt_inds = torch.cat(self.all_pos_assigned_gt_inds, -1)
                return targets

            return wrapper

        return wrapper_with_option
