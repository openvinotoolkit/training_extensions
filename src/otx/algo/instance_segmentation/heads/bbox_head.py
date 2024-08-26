# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.roi_heads.bbox_heads.bbox_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/roi_heads/bbox_heads/bbox_head.py
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
import torch.nn.functional
from torch import Tensor, nn
from torch.nn.modules.utils import _pair

from otx.algo.common.losses import CrossSigmoidFocalLoss
from otx.algo.common.utils.nms import multiclass_nms
from otx.algo.common.utils.structures import SamplingResult
from otx.algo.common.utils.utils import multi_apply
from otx.algo.detection.heads.class_incremental_mixin import (
    ClassIncrementalMixin,
)
from otx.algo.instance_segmentation.layers import multiclass_nms_torch
from otx.algo.instance_segmentation.losses import accuracy
from otx.algo.instance_segmentation.utils.structures.bbox import scale_boxes
from otx.algo.instance_segmentation.utils.utils import empty_instances
from otx.algo.modules.base_module import BaseModule
from otx.algo.utils.mmengine_utils import InstanceData

if TYPE_CHECKING:
    from otx.algo.common.utils.coders import DeltaXYWHBBoxCoder


class BBoxHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and regression respectively."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        bbox_coder: nn.Module,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
        roi_feat_size: int = 7,
        with_avg_pool: bool = False,
        with_cls: bool = True,
        with_reg: bool = True,
        predict_box_type: str = "hbox",
        reg_class_agnostic: bool = False,
        reg_decoded_bbox: bool = False,
        init_cfg: dict | list[dict] | None = None,
        pos_weight: float = -1,
        nms_iou_threshold: float = 0.5,
        max_per_img: int = 100,
        score_threshold: float = 0.05,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if not with_cls and not with_reg:
            msg = "with_cls and with_reg cannot be both False"
            raise ValueError(msg)
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
        self.pos_weight = pos_weight
        self.nms_iou_threshold = nms_iou_threshold
        self.max_per_img = max_per_img
        self.score_threshold = score_threshold

        self.bbox_coder = bbox_coder
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox

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

        Returns:
            tuple[Tensor]: Ground truth for proposals
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
            pos_weight = 1.0 if self.pos_weight <= 0 else self.pos_weight
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
        rescale: bool = False,
    ) -> list[InstanceData]:
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
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[InstanceData]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if len(cls_scores) != len(bbox_preds):
            msg = "The length of cls_scores and bbox_preds should be the same."
            raise ValueError(msg)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
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

        Returns:
            InstanceData: Detection results of each image\
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
                score_per_cls=False,
            )[0]

        scores = torch.nn.functional.softmax(cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta["img_shape"]
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        num_classes = 1 if self.reg_class_agnostic else self.num_classes
        roi = roi.repeat_interleave(num_classes, dim=0)
        bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
        bboxes = self.bbox_coder.decode(roi[..., 1:], bbox_pred, max_shape=img_shape)

        if rescale and bboxes.size(0) > 0:
            if img_meta.get("scale_factor") is None:
                msg = "scale_factor must be specified in img_meta"
                raise ValueError(msg)
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            bboxes = scale_boxes(bboxes, scale_factor)  # type: ignore [arg-type]

        # Get the inside tensor when `bboxes` is a box type
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        nms_cfg = {
            "type": "nms",
            "iou_threshold": self.nms_iou_threshold,
        }  # TODO(Kirill, Sungchul): depricate
        det_bboxes, det_labels = multiclass_nms_torch(  # type: ignore [misc]
            multi_bboxes=bboxes,
            multi_scores=scores,
            score_thr=self.score_threshold,
            nms_cfg=nms_cfg,
            max_num=self.max_per_img,
            box_dim=box_dim,
        )
        results.bboxes = det_bboxes[:, :-1]
        results.scores = det_bboxes[:, -1]
        results.labels = det_labels
        return results

    def export_by_feat(
        self,
        rois: Tensor,
        cls_scores: tuple[Tensor],
        bbox_preds: tuple[Tensor],
        batch_img_metas: list[dict],
        rcnn_test_cfg: dict,
        rescale: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Rewrite `predict_by_feat` of `BBoxHead` for default backend.

        Transform network output for a batch into bbox predictions. Support
        `reg_class_agnostic == False` case.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (dict, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
                - dets (Tensor): Classification bboxes and scores, has a shape
                    (num_instance, 5)
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
        """
        warnings.warn(f"rescale: {rescale} is not supported in ONNX export. Ignored.", stacklevel=2)
        if rois.ndim != 3:
            msg = "Only support export two stage model to ONNX with batch dimension."
            raise ValueError(msg)

        img_shape = batch_img_metas[0]["img_shape"]
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_scores)
        else:
            scores = torch.nn.functional.softmax(cls_scores, dim=-1) if cls_scores is not None else None

        if bbox_preds is not None:
            # num_classes = 1 if self.reg_class_agnostic else self.num_classes
            # if num_classes > 1:
            #     rois = rois.repeat_interleave(num_classes, dim=1)
            bboxes = self.bbox_coder.decode_export(rois[..., 1:], bbox_preds, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat([max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        batch_size = scores.shape[0]
        device = scores.device
        # ignore background class
        scores = scores[..., : self.num_classes]
        if not self.reg_class_agnostic:
            # only keep boxes with the max scores
            max_inds = scores.reshape(-1, self.num_classes).argmax(1, keepdim=True)
            encode_size = self.bbox_coder.encode_size
            bboxes = bboxes.reshape(-1, self.num_classes, encode_size)
            dim0_inds = torch.arange(bboxes.shape[0], device=device).unsqueeze(-1)
            bboxes = bboxes[dim0_inds, max_inds].reshape(batch_size, -1, encode_size)

        # get nms params
        max_output_boxes_per_class = 200
        pre_top_k = 5000
        iou_threshold = rcnn_test_cfg["nms"].get("iou_threshold")
        score_threshold = rcnn_test_cfg.get("score_thr", 0.05)
        keep_top_k = rcnn_test_cfg.get("max_per_img", 100)
        return multiclass_nms(
            bboxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k,
        )


class ConvFCBBoxHeadModule(BBoxHead, ClassIncrementalMixin):
    r"""More general bbox head, with shared conv and fc layers and two optional separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """

    def __init__(
        self,
        num_shared_convs: int = 0,
        num_shared_fcs: int = 2,
        num_cls_convs: int = 0,
        num_cls_fcs: int = 0,
        num_reg_convs: int = 0,
        num_reg_fcs: int = 0,
        conv_out_channels: int = 256,
        fc_out_channels: int = 1024,
        normalization: Callable[..., nn.Module] | None = None,
        init_cfg: dict | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)  # type: ignore [misc]
        if num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs + num_reg_convs + num_reg_fcs <= 0:
            msg = (
                "Pls specify at least one of num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, "
                "num_reg_convs, num_reg_fcs"
            )
            raise ValueError(msg)
        if (num_cls_convs > 0 or num_reg_convs > 0) and num_shared_fcs != 0:
            msg = "Shared FC layers are mutually exclusive with cls/reg conv layers"
            raise ValueError(msg)
        if (not self.with_cls) and (num_cls_convs != 0 or num_cls_fcs != 0):
            msg = "num_cls_convs and num_cls_fcs should be zero if without classification"
            raise ValueError(msg)
        if (not self.with_reg) and (num_reg_convs != 0 or num_reg_fcs != 0):
            msg = "num_reg_convs and num_reg_fcs should be zero if without regression"
            raise ValueError(msg)
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.normalization = normalization

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_shared_convs,
            self.num_shared_fcs,
            self.in_channels,
            True,
        )
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch(
            self.num_cls_convs,
            self.num_cls_fcs,
            self.shared_out_channels,
        )

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(
            self.num_reg_convs,
            self.num_reg_fcs,
            self.shared_out_channels,
        )

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            cls_channels = self.num_classes + 1
            self.fc_cls = nn.Linear(in_features=self.cls_last_dim, out_features=cls_channels)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else box_dim * self.num_classes
            self.fc_reg = nn.Linear(in_features=self.reg_last_dim, out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            if not isinstance(self.init_cfg, list):
                msg = "init_cfg must be a list"
                raise TypeError(msg)
            self.init_cfg += [
                {
                    "type": "Xavier",
                    "distribution": "uniform",
                    "override": [
                        {"name": "shared_fcs"},
                        {"name": "cls_fcs"},
                        {"name": "reg_fcs"},
                    ],
                },
            ]

    def _add_conv_fc_branch(
        self,
        num_branch_convs: int,
        num_branch_fcs: int,
        in_channels: int,
        is_shared: bool = False,
    ) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()

        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = last_layer_dim if i == 0 else self.fc_out_channels
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x: Tensor) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (Tensor): Features from the upstream network, each is a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part

        if self.num_shared_fcs > 0:
            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def loss_and_target(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        rois: Tensor,
        sampling_results: list[SamplingResult],
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
            sampling_results (list[SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_img_metas (list[dict]): Meta information of each image, e.g., image size, scaling factor, etc.
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
            batch_img_metas (list[dict]): Meta information of each image, e.g., image size, scaling factor, etc.
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


class ConvFCBBoxHead:
    """ConvFCBBoxHead factory for instance segmentation task."""

    BBOXHEAD_CFG: ClassVar[dict[str, Any]] = {
        "maskrcnn_resnet_50": {
            "in_channels": 256,
            "pos_weight": -1,
            "nms_iou_threshold": 0.5,
            "max_per_img": 100,
            "score_threshold": 0.05,
        },
        "maskrcnn_efficientnet_b2b": {
            "in_channels": 80,
            "pos_weight": -1,
            "nms_iou_threshold": 0.5,
            "max_per_img": 500,
            "score_threshold": 0.05,
        },
        "maskrcnn_swin_tiny": {
            "in_channels": 256,
            "pos_weight": -1,
            "nms_iou_threshold": 0.5,
            "max_per_img": 100,
            "score_threshold": 0.05,
        },
    }

    def __new__(
        cls,
        model_name: str,
        num_classes: int,
        bbox_coder: DeltaXYWHBBoxCoder,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
    ) -> ConvFCBBoxHeadModule:
        """RPNHead factory for instance segmentation regional proposal network."""
        return ConvFCBBoxHeadModule(
            **cls.BBOXHEAD_CFG[model_name],
            num_classes=num_classes,
            bbox_coder=bbox_coder,
            loss_bbox=loss_bbox,
            loss_cls=loss_cls,
        )
