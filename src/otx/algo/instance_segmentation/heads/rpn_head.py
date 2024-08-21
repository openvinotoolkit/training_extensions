# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.dense_heads.rpn_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/dense_heads/rpn_head.py
"""

from __future__ import annotations

import copy
import warnings

import torch
import torch.nn.functional
from torch import Tensor, nn

from otx.algo.common.utils.nms import batched_nms, multiclass_nms
from otx.algo.common.utils.utils import dynamic_topk, gather_topk
from otx.algo.detection.heads.anchor_head import AnchorHead
from otx.algo.instance_segmentation.utils.structures.bbox import empty_box_as, get_box_wh
from otx.algo.instance_segmentation.utils.utils import unpack_inst_seg_entity
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import OTXBatchDataEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity

# ruff: noqa: PLW2901


class RPNHead(AnchorHead):
    """Implementation of RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background
            category. Defaults to 1.
        init_cfg (dict or list[dict]): Initialization config dict.
        num_convs (int): Number of convolution layers in the head.
            Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        init_cfg: dict | list[dict] | None = None,
        num_convs: int = 1,
        **kwargs,
    ) -> None:
        self.num_convs = num_convs
        if init_cfg is None:
            init_cfg = {"type": "Normal", "layer": "Conv2d", "std": 0.01}

        if num_classes != 1:
            msg = "num_classes must be 1 for RPNHead"
            raise ValueError(msg)
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs,
        )

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                in_channels = self.in_channels if i == 0 else self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    Conv2dModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        activation=build_activation_layer(nn.ReLU, inplace=False),
                    ),
                )
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

    def loss_and_predict(
        self,
        x: tuple[Tensor],
        rpn_entity: InstanceSegBatchDataEntity,
        proposal_cfg: dict | None = None,
    ) -> tuple[dict, list[InstanceData]]:
        """Forward propagation of the head, then calculate loss and predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[InstanceSegBatchDataEntity]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (dict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[InstanceData]): Detection
                  results of each image after the post process.
        """
        batch_gt_instances, batch_img_metas = unpack_inst_seg_entity(rpn_entity)

        cls_scores, bbox_preds = self(x)

        losses = self.loss_by_feat(
            cls_scores,
            bbox_preds,
            batch_gt_instances,
            batch_img_metas,
        )

        predictions = self.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_img_metas=batch_img_metas,
            cfg=proposal_cfg,
        )
        return losses, predictions

    def predict(
        self,
        x: tuple[Tensor, ...],
        entity: OTXBatchDataEntity,
        rescale: bool = False,
    ) -> list[InstanceData]:
        """Forward-prop of the detection head and predict detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            entity (OTXBatchDataEntity): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[InstanceData]: Detection results of each image
            after the post process.
        """
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

        cls_scores, bbox_preds = self(x)

        return self.predict_by_feat(cls_scores, bbox_preds, batch_img_metas=batch_img_metas, rescale=rescale)

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_gt_instances: list[InstanceData],
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | None = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[InstanceData]): Batch of gt_instance.
                It usually includes ``bboxes`` and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[InstanceData], Optional):
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

    def _predict_by_feat_single(  # type: ignore[override]
        self,
        cls_score_list: list[Tensor],
        bbox_pred_list: list[Tensor],
        score_factor_list: list[Tensor],
        mlvl_priors: list[Tensor],
        img_meta: dict,
        cfg: dict,
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
            cfg (dict, optional): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            InstanceData: Detection results of each image
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
        cfg: dict,
        img_meta: dict,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceData:
        """Bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation.

        Args:
            results (InstaceData): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (dict): Test / postprocessing configuration.
            img_meta (dict, optional): Image meta info. Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.

        Returns:
            InstanceData: Detection results of each image
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
            w, h = get_box_wh(results.bboxes)  # type: ignore[attr-defined]
            valid_mask = (w > cfg["min_bbox_size"]) & (h > cfg["min_bbox_size"])
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:  # type: ignore[attr-defined]
            bboxes = results.bboxes  # type: ignore[attr-defined]
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.level_ids, cfg["nms"])  # type: ignore[attr-defined]
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[: cfg["max_per_img"]]

            #  in visualization
            results.labels = results.scores.new_zeros(len(results), dtype=torch.long)  # type: ignore[attr-defined]
            del results.level_ids  # type: ignore[attr-defined]
        else:
            # To avoid some potential error
            results_ = InstanceData()
            results_.bboxes = empty_box_as(results.bboxes)  # type: ignore[attr-defined]
            results_.scores = results.scores.new_zeros(0)  # type: ignore[attr-defined]
            results_.labels = results.scores.new_zeros(0)  # type: ignore[attr-defined]
            results = results_
        return results

    def export_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        score_factors: list[Tensor] | None = None,
        batch_img_metas: list[dict] | None = None,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rewrite `predict_by_feat` of `RPNHead` for default backend."""
        warnings.warn(f"score_factors: {score_factors} is not used in RPNHead.export", stacklevel=2)
        warnings.warn(f"rescale: {rescale} is not used in RPNHead.export", stacklevel=2)
        img_metas = batch_img_metas
        if len(cls_scores) != len(bbox_preds):
            msg = "cls_scores and bbox_preds should have the same length"
            raise ValueError(msg)
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
                "score_thr": 0.05,
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
            anchors = anchors.unsqueeze(0)

            if pre_topk > 0:
                _, topk_inds = dynamic_topk(scores.squeeze(2), pre_topk)
                bbox_pred, scores = gather_topk(
                    bbox_pred,
                    scores,
                    inds=topk_inds,
                    batch_size=batch_size,
                    is_batched=True,
                )
                anchors = gather_topk(
                    anchors,
                    inds=topk_inds,
                    batch_size=batch_size,
                    is_batched=False,
                )
            mlvl_valid_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_valid_anchors.append(anchors)

        batch_mlvl_bboxes = torch.cat(mlvl_valid_bboxes, dim=1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
        batch_mlvl_bboxes = self.bbox_coder.decode_export(
            batch_mlvl_anchors,
            batch_mlvl_bboxes,
            max_shape=img_metas[0]["img_shape"],  # type: ignore[index]
        )
        # ignore background class
        if not self.use_sigmoid_cls:
            batch_mlvl_scores = batch_mlvl_scores[..., : self.num_classes]
        if not with_nms:
            return batch_mlvl_bboxes, batch_mlvl_scores

        pre_top_k = 5000
        iou_threshold = cfg["nms"].get("iou_threshold")
        score_threshold = cfg.get("score_thr", 0.05)
        keep_top_k = cfg.get("max_per_img", 1000)
        # only one class in rpn
        max_output_boxes_per_class = keep_top_k
        return multiclass_nms(
            batch_mlvl_bboxes,
            batch_mlvl_scores,
            max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k,
        )
