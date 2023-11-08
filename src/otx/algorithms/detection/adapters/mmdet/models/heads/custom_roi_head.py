"""Custom ROI head for OTX template."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

from otx.algorithms.detection.adapters.mmdet.models.heads.cross_dataset_detector_head import (
    CrossDatasetDetectorHead,
)
from otx.algorithms.detection.adapters.mmdet.models.losses.cross_focal_loss import (
    CrossSigmoidFocalLoss,
)

# TODO: Need to fix pylint issues
# pylint: disable=abstract-method, arguments-renamed, too-many-locals, too-many-arguments


@HEADS.register_module()
class CustomRoIHead(StandardRoIHead):
    """CustomROIHead class for OTX."""

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``."""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        if bbox_head.type == "Shared2FCBBoxHead":
            bbox_head.type = "CustomConvFCBBoxHead"
        self.bbox_head = build_head(bbox_head)

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        labels, label_weights, bbox_targets, bbox_weights, valid_label_mask = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, img_metas, self.train_cfg
        )
        loss_bbox = self.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            valid_label_mask=valid_label_mask,
        )
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        mask_results = super()._mask_forward(x, rois, pos_inds, bbox_feats)
        if mask_results["mask_pred"].device.type == "hpu":
            mask_results["mask_pred"] = mask_results["mask_pred"].cpu()
            mask_results["mask_feats"] = mask_results["mask_feats"].cpu()
        return mask_results


@HEADS.register_module()
class CustomConvFCBBoxHead(Shared2FCBBoxHead, CrossDatasetDetectorHead):
    """CustomConvFCBBoxHead class for OTX."""

    def get_targets(self, sampling_results, gt_bboxes, gt_labels, img_metas, rcnn_train_cfg, concat=True):
        """Calculate the ground truth for all samples in a batch according to the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg,
        )
        valid_label_mask = self.get_valid_label_mask(img_metas=img_metas, all_labels=labels, use_bg=True)
        valid_label_mask = [i.to(gt_bboxes[0].device) for i in valid_label_mask]

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            valid_label_mask = torch.cat(valid_label_mask, 0)
        return labels, label_weights, bbox_targets, bbox_weights, valid_label_mask

    def forward(self, x):
        """ConvFCBBoxHead forward."""
        # shared part
        cls_score, bbox_pred = super().forward(x)
        if cls_score.device.type == "hpu":
            cls_score = cls_score.cpu()
            bbox_pred = bbox_pred.cpu()

        return cls_score, bbox_pred

    @force_fp32(apply_to=("cls_score", "bbox_pred"))
    def loss(
        self,
        cls_score,
        bbox_pred,
        rois,
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        reduction_override=None,
        valid_label_mask=None,
    ):
        """Loss function for CustomConvFCBBoxHead."""
        losses = dict()
        if cls_score is not None and cls_score.numel() > 0:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)

            if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
                losses["loss_cls"] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override,
                    use_bg=True,
                    valid_label_mask=valid_label_mask,
                )
            else:
                losses["loss_cls"] = self.loss_cls(
                    cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override=reduction_override
                )
            losses["acc"] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                        pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]
                    ]
                losses["loss_bbox"] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override,
                )
            else:
                losses["loss_bbox"] = bbox_pred[pos_inds].sum()
        return losses
