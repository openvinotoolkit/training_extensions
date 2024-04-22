# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet RPNHead."""
from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn.functional
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from torch import Tensor, nn

from otx.algo.detection.deployment import is_mmdeploy_enabled
from otx.algo.detection.heads.anchor_head import AnchorHead
from otx.algo.detection.ops.nms import batched_nms
from otx.algo.instance_segmentation.mmdet.structures.bbox import (
    empty_box_as,
    get_box_wh,
)
from otx.algo.modules.conv_module import ConvModule

# ruff: noqa: PLW2901

if TYPE_CHECKING:
    from mmengine.config import ConfigDict

    from otx.algo.instance_segmentation.mmdet.models.utils import InstanceList, OptInstanceList


@MODELS.register_module()
class RPNHead(AnchorHead):
    """Implementation of RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background
            category. Defaults to 1.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or \
            list[dict]): Initialization config dict.
        num_convs (int): Number of convolution layers in the head.
            Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        init_cfg: dict | None = None,
        num_convs: int = 1,
        **kwargs,
    ) -> None:
        self.num_convs = num_convs
        if init_cfg is None:
            init_cfg = {"type": "Normal", "layer": "Conv2d", "std": 0.01}

        if num_classes != 1:
            msg = "num_classes must be 1 for RPNHead"
            raise ValueError(msg)
        super().__init__(num_classes=num_classes, in_channels=in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                in_channels = self.in_channels if i == 0 else self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(ConvModule(in_channels, self.feat_channels, 3, padding=1, inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * reg_dim, 1)

    def forward_single(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        x = self.rpn_conv(x)
        x = torch.nn.functional.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[obj:InstanceData]): Batch of gt_instance.
                It usually includes ``bboxes`` and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[obj:InstanceData], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )
        return {"loss_rpn_cls": losses["loss_cls"], "loss_rpn_bbox": losses["loss_bbox"]}

    def _predict_by_feat_single(
        self,
        cls_score_list: list[Tensor],
        bbox_pred_list: list[Tensor],
        score_factor_list: list[Tensor],
        mlvl_priors: list[Tensor],
        img_meta: dict,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceData:
        """Transform a single image's features extracted from the head into bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Be compatible with
                BaseDenseHead. Not used in RPNHead.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (ConfigDict, optional): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta["img_shape"]
        nms_pre = cfg.get("nms_pre", -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        level_ids = []
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(zip(cls_score_list, bbox_pred_list, mlvl_priors)):
            if cls_score.size()[-2:] != bbox_pred.size()[-2:]:
                msg = "cls_score and bbox_pred should have the same size"
                raise RuntimeError(msg)

            reg_dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = cls_score.sigmoid() if self.use_sigmoid_cls else cls_score.softmax(-1)[:, :-1]

            scores = torch.squeeze(scores)
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)

            # use level id to implement the separate level nms
            level_ids.append(scores.new_full((scores.size(0),), level_idx, dtype=torch.long))

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.level_ids = torch.cat(level_ids)

        return self._bbox_post_process(results=results, cfg=cfg, rescale=rescale, img_meta=img_meta)

    def _bbox_post_process(
        self,
        results: InstanceData,
        cfg: ConfigDict,
        img_meta: dict,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceData:
        """Bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if not with_nms:
            msg = "`with_nms` must be True in RPNHead"
            raise RuntimeError(msg)

        if rescale:
            msg = "Rescale is not implemented in RPNHead"
            raise NotImplementedError

        # filter small size bboxes
        if cfg.get("min_bbox_size", -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = results.bboxes
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.level_ids, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[: cfg.max_per_img]

            #  in visualization
            results.labels = results.scores.new_zeros(len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            results_.bboxes = empty_box_as(results.bboxes)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_
        return results


if is_mmdeploy_enabled():
    from mmdeploy.codebase.mmdet.deploy import gather_topk, get_post_processing_params, pad_with_value_if_necessary
    from mmdeploy.core import FUNCTION_REWRITER
    from mmdeploy.mmcv.ops import multiclass_nms as multiclass_nms_ops
    from mmdeploy.utils import is_dynamic_shape

    @FUNCTION_REWRITER.register_rewriter(
        func_name="otx.algo.instance_segmentation.mmdet.models.dense_heads.rpn_head.RPNHead.predict_by_feat",
    )
    def rpn_head__predict_by_feat(
        self: RPNHead,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_img_metas: list[dict],
        score_factors: list[Tensor] | None = None,
        cfg: ConfigDict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
        **kwargs,
    ) -> tuple:
        """Rewrite `predict_by_feat` of `RPNHead` for default backend.

        Rewrite this function to deploy model, transform network output for a
        batch into bbox predictions.

        Args:
            ctx (ContextCaller): The context with additional information.
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            If with_nms == True:
                tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
                `dets` of shape [N, num_det, 5] and `labels` of shape
                [N, num_det].
            Else:
                tuple[Tensor, Tensor, Tensor]: batch_mlvl_bboxes,
                    batch_mlvl_scores, batch_mlvl_centerness
        """
        warnings.warn(f"score_factors: {score_factors} is not used in RPNHead", stacklevel=2)
        warnings.warn(f"rescale: {rescale} is not used in RPNHead", stacklevel=2)
        warnings.warn(f"kwargs: {kwargs} is not used in RPNHead", stacklevel=2)
        ctx = FUNCTION_REWRITER.get_context()
        img_metas = batch_img_metas
        if len(cls_scores) != len(bbox_preds):
            msg = "cls_scores and bbox_preds should have the same length"
            raise ValueError(msg)
        deploy_cfg = ctx.cfg
        is_dynamic_flag = is_dynamic_shape(deploy_cfg)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
        if len(mlvl_cls_scores) != len(mlvl_bbox_preds) != len(mlvl_anchors):
            msg = "mlvl_cls_scores, mlvl_bbox_preds and mlvl_anchors should have the same length"
            raise ValueError(msg)

        cfg = self.test_cfg if cfg is None else cfg
        if cfg is None:
            warnings.warn("cfg is None, use default cfg", stacklevel=2)
            cfg = {
                "max_per_img": 1000,
                "min_bbox_size": 0,
                "nms": {"iou_threshold": 0.7, "type": "nms"},
                "nms_pre": 1000,
            }
        batch_size = mlvl_cls_scores[0].shape[0]
        pre_topk = cfg.get("nms_pre", -1)

        # loop over features, decode boxes
        mlvl_valid_bboxes = []
        mlvl_scores = []
        mlvl_valid_anchors = []
        for cls_score, bbox_pred, anchors in zip(
            mlvl_cls_scores,
            mlvl_bbox_preds,
            mlvl_anchors,
        ):
            if cls_score.size()[-2:] != bbox_pred.size()[-2:]:
                msg = "cls_score and bbox_pred should have the same size"
                raise ValueError(msg)
            cls_score = cls_score.permute(0, 2, 3, 1)
            if self.use_sigmoid_cls:
                cls_score = cls_score.reshape(batch_size, -1)
                scores = cls_score.sigmoid()
            else:
                cls_score = cls_score.reshape(batch_size, -1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = cls_score.softmax(-1)[..., 0]
            scores = scores.reshape(batch_size, -1, 1)
            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, dim)

            # use static anchor if input shape is static
            if not is_dynamic_flag:
                anchors = anchors.data

            anchors = anchors.unsqueeze(0)

            # topk in tensorrt does not support shape<k
            # concate zero to enable topk,
            scores = pad_with_value_if_necessary(scores, 1, pre_topk, 0.0)
            bbox_pred = pad_with_value_if_necessary(bbox_pred, 1, pre_topk)
            anchors = pad_with_value_if_necessary(anchors, 1, pre_topk)

            if pre_topk > 0:
                _, topk_inds = scores.squeeze(2).topk(pre_topk)
                bbox_pred, scores = gather_topk(
                    bbox_pred,
                    scores,
                    inds=topk_inds,
                    batch_size=batch_size,
                    is_batched=True,
                )
                anchors = gather_topk(anchors, inds=topk_inds, batch_size=batch_size, is_batched=False)
            mlvl_valid_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_valid_anchors.append(anchors)

        batch_mlvl_bboxes = torch.cat(mlvl_valid_bboxes, dim=1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
        batch_mlvl_bboxes = self.bbox_coder.decode(
            batch_mlvl_anchors,
            batch_mlvl_bboxes,
            max_shape=img_metas[0]["img_shape"],
        )
        # ignore background class
        if not self.use_sigmoid_cls:
            batch_mlvl_scores = batch_mlvl_scores[..., : self.num_classes]
        if not with_nms:
            return batch_mlvl_bboxes, batch_mlvl_scores

        post_params = get_post_processing_params(deploy_cfg)
        iou_threshold = cfg["nms"].get("iou_threshold", post_params.iou_threshold)
        score_threshold = cfg.get("score_thr", post_params.score_threshold)
        pre_top_k = post_params.pre_top_k
        keep_top_k = cfg.get("max_per_img", post_params.keep_top_k)
        # only one class in rpn
        max_output_boxes_per_class = keep_top_k
        nms_type = cfg["nms"].get("type")
        return multiclass_nms_ops(
            batch_mlvl_bboxes,
            batch_mlvl_scores,
            max_output_boxes_per_class,
            nms_type=nms_type,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k,
        )
