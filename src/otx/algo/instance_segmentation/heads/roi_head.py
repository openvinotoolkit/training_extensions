# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from otx.algo.instance_segmentation.utils.structures.bbox import bbox2roi
from otx.algo.instance_segmentation.utils.utils import empty_instances, unpack_inst_seg_entity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity

if TYPE_CHECKING:
    from otx.algo.utils.mmengine_utils import InstanceData


class RoIHead(nn.Module):
    """Base class for RoIHeads.

    Args:
        bbox_roi_extractor (nn.Module): Module to extract bbox features.
        bbox_head (nn.Module): Module to make predictions from bbox features.
        mask_roi_extractor (nn.Module): Module to extract mask features.
        mask_head (nn.Module): Module to make predictions from mask features.
        assigner (nn.Module): Module to assign gt to bboxes.
        sampler (nn.Module): Module to sample bboxes.
        mask_thr_binary (float): Threshold to convert mask to binary.
        max_per_img (int): Maximum number of instances per image.
        nms_iou_threshold (float): IoU threshold for NMS.
        score_thr (float): Threshold to filter out low score
    """

    def __init__(
        self,
        bbox_roi_extractor: nn.Module,
        bbox_head: nn.Module,
        mask_roi_extractor: nn.Module,
        mask_head: nn.Module,
        assigner: nn.Module,
        sampler: nn.Module,
        mask_thr_binary: float = 0.5,
        max_per_img: int = 100,
        nms_iou_threshold: float = 0.5,
        score_thr: float = 0.05,
    ) -> None:
        super().__init__()

        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_head = bbox_head
        self.mask_thr_binary = mask_thr_binary
        self.max_per_img = max_per_img
        self.nms_iou_threshold = nms_iou_threshold
        self.score_thr = score_thr

        self.mask_roi_extractor = mask_roi_extractor
        self.mask_head = mask_head
        self.bbox_assigner = assigner
        self.bbox_sampler = sampler

    @property
    def with_bbox(self) -> bool:
        """bool: whether the RoI head contains a `bbox_head`."""
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_mask(self) -> bool:
        """bool: whether the RoI head contains a `mask_head`."""
        return hasattr(self, "mask_head") and self.mask_head is not None

    @property
    def with_shared_head(self) -> bool:
        """bool: whether the RoI head contains a `shared_head`."""
        return hasattr(self, "shared_head") and self.shared_head is not None

    def _bbox_forward(self, x: tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        bbox_feats = self.bbox_roi_extractor(x[: self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        return {"cls_score": cls_score, "bbox_pred": bbox_pred, "bbox_feats": bbox_feats}

    def _mask_forward(
        self,
        x: tuple[Tensor],
        rois: Tensor | None = None,
        pos_inds: Tensor | None = None,
        bbox_feats: Tensor | None = None,
    ) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        if not ((rois is not None) ^ (pos_inds is not None and bbox_feats is not None)):
            msg = "rois is None xor (pos_inds is not None and bbox_feats is not None)"
            raise ValueError(msg)
        if rois is not None:
            mask_feats = self.mask_roi_extractor(x[: self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            if bbox_feats is None:
                msg = "bbox_feats should not be None when rois is None"
                raise ValueError(msg)
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats)
        return {"mask_preds": mask_preds, "mask_feats": mask_feats}

    def predict_bbox(
        self,
        x: tuple[Tensor],
        batch_img_metas: list[dict],
        rpn_results_list: list[InstanceData],
        rescale: bool = False,
    ) -> list[InstanceData]:
        """Forward the bbox head and predict detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[InstanceData]): List of region
                proposals.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[InstanceData]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]  # type: ignore[attr-defined]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type="bbox",
                num_classes=self.bbox_head.num_classes,
                score_per_cls=False,
            )

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results["cls_score"]
        bbox_preds = bbox_results["bbox_pred"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None,) * len(proposals)

        return self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
        )

    def predict_mask(
        self,
        x: tuple[Tensor],
        batch_img_metas: list[dict],
        results_list: list[InstanceData],
        rescale: bool = False,
    ) -> list[InstanceData]:
        """Forward the mask head and predict detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[InstanceData]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[InstanceData]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]  # type: ignore[attr-defined]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.mask_thr_binary,
            )

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results["mask_preds"]
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        return self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
        )

    def _bbox_forward_export(self, x: tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        bbox_feats = self.bbox_roi_extractor.export(
            x[: self.bbox_roi_extractor.num_inputs],
            rois,
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        return {"cls_score": cls_score, "bbox_pred": bbox_pred, "bbox_feats": bbox_feats}

    def _mask_forward_export(
        self,
        x: tuple[Tensor],
        rois: Tensor | None = None,
        pos_inds: Tensor | None = None,
        bbox_feats: Tensor | None = None,
    ) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        if not ((rois is not None) ^ (pos_inds is not None and bbox_feats is not None)):
            msg = "rois is None xor (pos_inds is not None and bbox_feats is not None)"
            raise ValueError(msg)
        if rois is not None:
            mask_feats = self.mask_roi_extractor.export(x[: self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            if bbox_feats is None:
                msg = "bbox_feats should not be None when rois is None"
                raise ValueError(msg)
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats)
        return {"mask_preds": mask_preds, "mask_feats": mask_feats}

    def predict(
        self,
        x: tuple[Tensor],
        rpn_results_list: list[InstanceData],
        entity: InstanceSegBatchDataEntity,
        rescale: bool = False,
    ) -> list[InstanceData]:
        """Forward the roi head and predict detection results on the features of the upstream network."""
        if not self.with_bbox:
            msg = "Bbox head must be implemented."
            raise NotImplementedError(msg)
        batch_img_metas = [
            {
                "img_id": img_info.img_idx,
                "img_shape": img_info.img_shape,
                "ori_shape": img_info.ori_shape,
                "scale_factor": img_info.scale_factor,
                "ignored_labels": img_info.ignored_labels,
            }
            for img_info in entity.imgs_info
        ]

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rescale=bbox_rescale,
        )

        if self.with_mask:
            results_list = self.predict_mask(x, batch_img_metas, results_list, rescale=rescale)

        return results_list

    def export(
        self,
        x: tuple[Tensor],
        rpn_results_list: tuple[Tensor, Tensor],
        batch_img_metas: list[dict],
        rescale: bool = False,
    ) -> tuple[Tensor, ...]:
        """Export the roi head and export detection results on the features of the upstream network."""
        if not self.with_bbox:
            msg = "Bbox head must be implemented."
            raise NotImplementedError(msg)

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.export_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rescale=bbox_rescale,
        )

        if self.with_mask:
            results_list = self.export_mask(x, batch_img_metas, results_list, rescale=rescale)

        return results_list

    def export_bbox(
        self,
        x: tuple[Tensor],
        batch_img_metas: list[dict],
        rpn_results_list: tuple[Tensor, Tensor],
        rescale: bool = False,
    ) -> tuple[Tensor, ...]:
        """Rewrite `predict_bbox` of `RoIHead` for default backend.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[Tensor]): List of region
                proposals.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[Tensor]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - dets (Tensor): Classification bboxes and scores, has a shape
                    (num_instance, 5)
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
        """
        rois = rpn_results_list[0]
        rois_dims = int(rois.shape[-1])
        batch_index = (
            torch.arange(rois.shape[0], device=rois.device).float().view(-1, 1, 1).expand(rois.size(0), rois.size(1), 1)
        )
        rois = torch.cat([batch_index, rois[..., : rois_dims - 1]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, rois_dims)
        bbox_results = self._bbox_forward_export(x, rois)
        cls_scores = bbox_results["cls_score"]
        bbox_preds = bbox_results["bbox_pred"]

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_scores = cls_scores.reshape(batch_size, num_proposals_per_img, cls_scores.size(-1))
        bbox_preds = bbox_preds.reshape(batch_size, num_proposals_per_img, bbox_preds.size(-1))

        return self.bbox_head.export_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
        )

    def export_mask(
        self,
        x: tuple[Tensor],
        batch_img_metas: list[dict],
        results_list: tuple[Tensor, ...],
        rescale: bool = False,
    ) -> tuple[Tensor, ...]:
        """Forward the mask head and predict detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[InstanceData]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[Tensor]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        dets, det_labels = results_list
        batch_size = dets.size(0)
        det_bboxes = dets[..., :4]
        # expand might lead to static shape, use broadcast instead
        batch_index = torch.arange(det_bboxes.size(0), device=det_bboxes.device).float().view(
            -1,
            1,
            1,
        ) + det_bboxes.new_zeros((det_bboxes.size(0), det_bboxes.size(1))).unsqueeze(-1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward_export(x, mask_rois)
        mask_preds = mask_results["mask_preds"]
        num_det = det_bboxes.shape[1]
        segm_results: Tensor = self.mask_head.export_by_feat(
            mask_preds,
            results_list,
            batch_img_metas,
            rescale=rescale,
        )
        segm_results = segm_results.reshape(batch_size, num_det, segm_results.shape[-2], segm_results.shape[-1])
        return dets, det_labels, segm_results

    def prepare_loss_inputs(
        self,
        x: tuple[Tensor],
        rpn_results_list: list[InstanceData],
        entity: InstanceSegBatchDataEntity,
    ) -> tuple[dict, dict, Any, Any, Any]:
        """Perform forward propagation and prepare outputs for loss calculation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            rpn_results_list (list[InstanceData]): List of region proposals.
            entity (InstanceSegBatchDataEntity): Entity from OTX dataset.

        Returns:
            dict: A dictionary of components for loss calculation.
        """
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

        # bbox head loss
        rois = bbox2roi([res.bboxes for res in sampling_results])
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)
        cls_reg_targets = self.bbox_head.get_targets(
            sampling_results,
            concat=True,
            batch_img_metas=batch_img_metas,
        )

        # mask head forward and loss
        mask_results = self._mask_forward(x, pos_rois)
        mask_targets = self.mask_head.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
        )
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        return bbox_results, mask_results, cls_reg_targets, mask_targets, pos_labels
