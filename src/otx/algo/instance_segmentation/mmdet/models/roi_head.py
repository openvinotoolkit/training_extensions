"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mmengine.registry import MODELS, TASK_UTILS
from torch import Tensor

from otx.algo.detection.heads.class_incremental_mixin import (
    ClassIncrementalMixin,
)
from otx.algo.detection.losses import CrossSigmoidFocalLoss, accuracy
from otx.algo.instance_segmentation.mmdet.models.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from otx.algo.instance_segmentation.mmdet.models.mask_heads.fcn_mask_head import FCNMaskHead
from otx.algo.instance_segmentation.mmdet.models.samplers import SamplingResult
from otx.algo.instance_segmentation.mmdet.models.utils import (
    ConfigType,
    InstanceList,
    empty_instances,
    multi_apply,
    unpack_gt_instances,
)
from otx.algo.instance_segmentation.mmdet.structures import DetDataSample
from otx.algo.instance_segmentation.mmdet.structures.bbox import bbox2roi

from .base_roi_head import BaseRoIHead
from .roi_extractors import SingleRoIExtractor

if TYPE_CHECKING:
    from mmengine.config import ConfigDict

    from otx.algo.instance_segmentation.mmdet.structures import SampleList


@MODELS.register_module()
class StandardRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = TASK_UTILS.build(self.train_cfg["assigner"])
        self.bbox_sampler = TASK_UTILS.build(self.train_cfg["sampler"], default_args={"context": self})

    def init_bbox_head(self, bbox_roi_extractor: ConfigType, bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        if bbox_roi_extractor["type"] != SingleRoIExtractor.__name__:
            msg = f"bbox_roi_extractor should be SingleRoIExtractor, but got {bbox_roi_extractor['type']}"
            raise ValueError(msg)

        if bbox_head["type"] != CustomConvFCBBoxHead.__name__:
            msg = f"bbox_head should be CustomConvFCBBoxHead, but got {bbox_head['type']}"
            raise ValueError(msg)

        bbox_roi_extractor.pop("type")
        bbox_head.pop("type")

        self.bbox_roi_extractor = SingleRoIExtractor(**bbox_roi_extractor)
        self.bbox_head = CustomConvFCBBoxHead(**bbox_head)

    def init_mask_head(self, mask_roi_extractor: ConfigType, mask_head: ConfigType) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict or ConfigDict): Config of mask roi
                extractor.
            mask_head (dict or ConfigDict): Config of mask in mask head.
        """
        if mask_roi_extractor["type"] != SingleRoIExtractor.__name__:
            msg = f"mask_roi_extractor should be SingleRoIExtractor, but got {mask_roi_extractor['type']}"
            raise ValueError(msg)
        mask_roi_extractor.pop("type")
        self.mask_roi_extractor = SingleRoIExtractor(**mask_roi_extractor)

        if mask_head["type"] != FCNMaskHead.__name__:
            msg = f"mask_head should be FCNMaskHead, but got {mask_head['type']}"
            raise ValueError(msg)

        mask_head.pop("type")
        self.mask_head = FCNMaskHead(**mask_head)

    def forward(
        self,
        x: tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: SampleList | None = None,
    ) -> tuple:
        """Network forward process. Usually includes backbone, neck and head forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
            the meta information of each image and corresponding
            annotations.

        Returns:
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        raise NotImplementedError

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

    def mask_loss(
        self,
        x: tuple[Tensor],
        sampling_results: list[SamplingResult],
        bbox_feats: Tensor,
        batch_gt_instances: InstanceList,
    ) -> dict:
        """Perform forward propagation and loss calculation of the mask head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
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
            rcnn_train_cfg=self.train_cfg,
        )

        mask_results.update(loss_mask=mask_loss_and_target["loss_mask"])
        return mask_results

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
        rpn_results_list: InstanceList,
        rcnn_test_cfg: ConfigType,
        rescale: bool = False,
    ) -> InstanceList:
        """Forward the bbox head and predict detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type="bbox",
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None,
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
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale,
        )

    def predict_mask(
        self,
        x: tuple[Tensor],
        batch_img_metas: list[dict],
        results_list: InstanceList,
        rescale: bool = False,
    ) -> InstanceList:
        """Forward the mask head and predict detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
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
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.test_cfg["mask_thr_binary"],
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
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
        )


@MODELS.register_module()
class CustomRoIHead(StandardRoIHead):
    """CustomRoIHead class for OTX."""

    def loss(self, x: tuple[Tensor], rpn_results_list: InstanceList, batch_data_samples: list[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection roi on the features.

        Args:
            x (tuple[Tensor]): list of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop("bboxes")

            assign_result = self.bbox_assigner.assign(rpn_results, batch_gt_instances[i], batch_gt_instances_ignore[i])
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
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            batch_img_metas (list[Dict]): Meta information of each image, e.g., image size, scaling factor, etc.

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
            rcnn_train_cfg=self.train_cfg,
            batch_img_metas=batch_img_metas,
        )
        bbox_results.update(loss_bbox=bbox_loss_and_target["loss_bbox"])

        return bbox_results


@MODELS.register_module()
class CustomConvFCBBoxHead(Shared2FCBBoxHead, ClassIncrementalMixin):
    """CustomConvFCBBoxHead class for OTX."""

    def loss_and_target(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        rois: Tensor,
        sampling_results: list[SamplingResult],
        rcnn_train_cfg: ConfigDict,
        batch_img_metas: list[dict],
        concat: bool = True,
        reduction_override: str | None = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (list[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            batch_img_metas (list[Dict]): Meta information of each image, e.g., image size, scaling factor, etc.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """
        cls_reg_targets = self.get_targets(
            sampling_results,
            rcnn_train_cfg,
            concat=concat,
            batch_img_metas=batch_img_metas,
        )
        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override,  # type: ignore[misc]
        )

        # cls_reg_targets is only for cascade rcnn
        return {"loss_bbox": losses, "bbox_targets": cls_reg_targets}

    def get_targets(
        self,
        sampling_results: list[SamplingResult],
        rcnn_train_cfg: ConfigDict,
        batch_img_metas: list[dict],
        concat: bool = True,
    ) -> tuple:
        """Calculate the ground truth for all samples in a batch according to the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (list[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            batch_img_metas (list[Dict]): Meta information of each image, e.g., image size, scaling factor, etc.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            tuple[Tensor]: Ground truth for proposals in a single image.
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
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_targets_single,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg,
        )

        valid_label_mask = self.get_valid_label_mask(img_metas=batch_img_metas, all_labels=labels, use_bg=True)
        valid_label_mask = [i.to(labels[0].device) for i in valid_label_mask]

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            valid_label_mask = torch.cat(valid_label_mask, 0)
        return labels, label_weights, bbox_targets, bbox_weights, valid_label_mask

    def loss(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        rois: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        valid_label_mask: Tensor | None = None,
        reduction_override: str | None = None,
    ) -> dict:
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
                if self.reg_class_agnostic:
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
        return losses
