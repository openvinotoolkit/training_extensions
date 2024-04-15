"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.registry import MODELS, TASK_UTILS
from mmengine.structures import InstanceData
from torch import Tensor, nn
from torch.nn.modules.utils import _pair

from otx.algo.instance_segmentation.mmdet.models.layers import multiclass_nms
from otx.algo.instance_segmentation.mmdet.models.utils import (
    InstanceList,
    OptMultiConfig,
    empty_instances,
)
from otx.algo.instance_segmentation.mmdet.structures.bbox import scale_boxes


class BBoxHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and regression respectively."""

    def __init__(
        self,
        in_channels: int,
        roi_feat_size: int,
        num_classes: int,
        bbox_coder: dict,
        loss_cls: dict,
        loss_bbox: dict,
        with_avg_pool: bool = False,
        with_cls: bool = True,
        with_reg: bool = True,
        predict_box_type: str = "hbox",
        reg_class_agnostic: bool = False,
        reg_decoded_bbox: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.predict_box_type = predict_box_type
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            cls_channels = num_classes + 1
            self.fc_cls = nn.Linear(in_features=in_channels, out_features=cls_channels)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if reg_class_agnostic else box_dim * num_classes
            self.fc_reg = nn.Linear(in_features=in_channels, out_features=out_dim_reg)
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            if self.with_cls:
                self.init_cfg += [
                    {
                        "type": "Normal",
                        "std": 0.01,
                        "override": {"name": "fc_cls"},
                    },
                ]
            if self.with_reg:
                self.init_cfg += [
                    {
                        "type": "Normal",
                        "std": 0.001,
                        "override": {"name": "fc_reg"},
                    },
                ]

    @property
    def custom_cls_channels(self) -> bool:
        """Get custom_cls_channels from loss_cls."""
        return getattr(self.loss_cls, "custom_cls_channels", False)

    def _get_targets_single(
        self,
        pos_priors: Tensor,
        neg_priors: Tensor,
        pos_gt_bboxes: Tensor,
        pos_gt_labels: Tensor,
        cfg: ConfigDict,
    ) -> tuple:
        """Calculate the ground truth for proposals in the single image according to the sampling results.

        Args:
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples,), self.num_classes, dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_priors, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def predict_by_feat(
        self,
        rois: tuple[Tensor],
        cls_scores: tuple[Tensor],
        bbox_preds: tuple[Tensor],
        batch_img_metas: list[dict],
        rcnn_test_cfg: ConfigDict | None = None,
        rescale: bool = False,
    ) -> InstanceList:
        """Transform a batch of output features extracted from the head into bbox results.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg,
            )
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(
        self,
        roi: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        img_meta: dict,
        rescale: bool = False,
        rcnn_test_cfg: ConfigDict | None = None,
    ) -> InstanceData:
        """Transform a single image's features extracted from the head into bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances(
                [img_meta],
                roi.device,
                task_type="bbox",
                instance_results=[results],
                num_classes=self.num_classes,
                score_per_cls=rcnn_test_cfg is None,
            )[0]

        scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta["img_shape"]
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        num_classes = 1 if self.reg_class_agnostic else self.num_classes
        roi = roi.repeat_interleave(num_classes, dim=0)
        bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
        bboxes = self.bbox_coder.decode(roi[..., 1:], bbox_pred, max_shape=img_shape)

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get("scale_factor") is not None
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        det_bboxes, det_labels = multiclass_nms(
            bboxes,
            scores,
            rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms,
            rcnn_test_cfg.max_per_img,
            box_dim=box_dim,
        )
        results.bboxes = det_bboxes[:, :-1]
        results.scores = det_bboxes[:, -1]
        results.labels = det_labels
        return results
