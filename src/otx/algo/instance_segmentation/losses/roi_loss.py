# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""ATSS criterion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from otx.algo.common.losses import CrossSigmoidFocalLoss
from otx.algo.instance_segmentation.losses import accuracy


class ROICriterion(nn.Module):
    """ROICriterion is a loss criterion used in the Region of Interest (ROI) algorithm.

    Args:
        num_classes (int): The number of object classes.
        bbox_coder (nn.Module): The module used for encoding and decoding bounding box coordinates.
        loss_cls (nn.Module): The module used for calculating the classification loss.
        loss_bbox (nn.Module): The module used for calculating the bounding box regression loss.
        loss_centerness (nn.Module | None, optional): The module used for calculating the centerness loss.
            Defaults to None.
        use_qfl (bool, optional): Whether to use the Quality Focal Loss (QFL).
            Defaults to ``CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)``.
        reg_decoded_bbox (bool, optional): Whether to use the decoded bounding box coordinates
            for regression loss calculation. Defaults to True.
        bg_loss_weight (float, optional): The weight for the background loss.
            Defaults to -1.0.
    """

    def __init__(
        self,
        num_classes: int,
        bbox_coder: nn.Module,
        loss_cls: nn.Module,
        loss_mask: nn.Module,
        loss_bbox: nn.Module,
        class_agnostic: bool = False,
        reg_decoded_bbox: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bbox_coder = bbox_coder
        self.loss_bbox = loss_bbox
        self.loss_cls = loss_cls
        self.loss_mask = loss_mask
        self.use_sigmoid_cls = loss_cls.use_sigmoid
        self.class_agnostic = class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox

        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            msg = f"num_classes={num_classes} is too small"
            raise ValueError(msg)

    def forward(
        self,
        rois: Tensor,
        pos_labels: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        mask_preds: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        mask_targets: Tensor,
        valid_label_mask: Tensor,
        avg_factor: float,
        reduction_override: str | None = None,
    ) -> dict[str, Tensor]:
        """Loss function for CustomConvFCBBoxHead."""
        losses = {}
        if cls_score is not None and cls_score.numel() > 0:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)

            if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
                losses["loss_cls"] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override,
                    valid_label_mask=valid_label_mask,
                )
            else:
                losses["loss_cls"] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override,
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
                if self.class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                        pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)],
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

        if mask_preds is not None:
            if mask_preds.size(0) == 0:
                loss_mask = mask_preds.sum()
            elif self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)

            losses["loss_mask"] = loss_mask

        return losses

    def loss(
        self,
        x: tuple[Tensor],
        rpn_results_list: list[InstanceData],
        entity: InstanceSegBatchDataEntity,
    ) -> dict:
        """Perform forward propagation and loss calculation of the detection roi on the features."""
        batch_gt_instances, batch_img_metas = unpack_inst_seg_entity(entity)

        # assign gts and sample proposals
        num_imgs = entity.batch_size
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop("bboxes")

            assign_result = self.bbox_assigner.assign(rpn_results, batch_gt_instances[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x],
            )
            sampling_results.append(sampling_result)

        losses = {}
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results, batch_img_metas)
            losses.update(bbox_results["loss_bbox"])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results, bbox_results["bbox_feats"], batch_gt_instances)
            losses.update(mask_results["loss_mask"])

        return losses

    def bbox_loss(self, x: tuple[Tensor], sampling_results: list[SamplingResult], batch_img_metas: list[dict]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): list of multi-level img features.
            sampling_results (list[SamplingResult]): Sampling results.
            batch_img_metas (list[dict]): Meta information of each image, e.g., image size, scaling factor, etc.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results["cls_score"],
            bbox_pred=bbox_results["bbox_pred"],
            rois=rois,
            sampling_results=sampling_results,
            batch_img_metas=batch_img_metas,
        )
        bbox_results.update(loss_bbox=bbox_loss_and_target["loss_bbox"])

        return bbox_results

    def mask_loss(
        self,
        x: tuple[Tensor],
        sampling_results: list[SamplingResult],
        bbox_feats: Tensor,
        batch_gt_instances: list[InstanceData],
    ) -> dict:
        """Perform forward propagation and loss calculation of the mask head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list[SamplingResult]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(x, pos_rois)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
        )

        mask_results.update(loss_mask=mask_loss_and_target["loss_mask"])
        return mask_results
