# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.dense_heads.rpn_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/dense_heads/rpn_head.py
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional
from torch import Tensor, nn

from otx.algo.common.utils.nms import batched_nms, multiclass_nms
from otx.algo.common.utils.utils import dynamic_topk, gather_topk
from otx.algo.detection.heads.anchor_head import AnchorHead
from otx.algo.instance_segmentation.utils.structures.bbox import empty_box_as, get_box_wh
from otx.algo.instance_segmentation.utils.utils import unpack_inst_seg_entity
from otx.algo.modules import build_activation_layer
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import OTXBatchDataEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity

if TYPE_CHECKING:
    from otx.algo.common.utils.assigners import MaxIoUAssigner
    from otx.algo.common.utils.coders import DeltaXYWHBBoxCoder
    from otx.algo.common.utils.prior_generators import AnchorGenerator
    from otx.algo.common.utils.samplers.base_sampler import BaseSampler

# ruff: noqa: PLW2901


class RPNHeadModule(AnchorHead):
    """Implementation of RPN head.

    Args:
        in_channels (int): Number of input channels.
        anchor_generator (nn.Module): Module that generates anchors.
        bbox_coder (nn.Module): Module that encodes/decodes bboxes.
        assigner (nn.Module): Module that assigns bboxes to ground truth.
        sampler (nn.Module): Module that samples bboxes.
        train_cfg (dict): Training configuration.
        test_cfg (dict): Testing configuration.
        init_cfg (dict, optional): Initialization configuration. Defaults to None.
        feat_channels (int, optional): Number of feature channels. Defaults to 256.
        reg_decoded_bbox (bool, optional): Whether to decode bbox. Defaults to False.
        allowed_border (float, optional): Allowed border. Defaults to 0.0.
        pos_weight (float, optional): Positive weight. Defaults to 1.0.
        num_classes (int, optional): Number of classes. Defaults to 1.
        num_convs (int, optional): Number of convolutions. Defaults to 1.
        max_per_img (int, optional): Maximum number of bboxes per image. Defaults to 1000.
        min_bbox_size (int, optional): Minimum bbox size. Defaults to 0.
        nms_iou_threshold (float, optional): NMS IoU threshold. Defaults to 0.7.
        score_threshold (float, optional): Score threshold. Defaults to 0.5.
        nms_pre (int, optional): NMS pre. Defaults to 1000.
        with_nms (bool, optional): Whether to use NMS. Defaults to True.
        use_sigmoid_cls (bool): Whether to use a sigmoid activation function
            for classification prediction. Defaults to True.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        anchor_generator: nn.Module,
        bbox_coder: nn.Module,
        assigner: nn.Module,
        sampler: nn.Module,
        train_cfg: dict,
        test_cfg: dict,
        init_cfg: dict | None = None,
        feat_channels: int = 256,
        reg_decoded_bbox: bool = False,
        allowed_border: float = 0.0,
        pos_weight: float = 1.0,
        num_classes: int = 1,
        num_convs: int = 1,
        max_per_img: int = 1000,
        min_bbox_size: int = 0,
        nms_iou_threshold: float = 0.7,
        score_threshold: float = 0.5,
        nms_pre: int = 1000,
        with_nms: bool = True,
        use_sigmoid_cls: bool = True,
    ) -> None:
        self.num_convs = num_convs
        if num_classes != 1:
            msg = "num_classes must be 1 for RPNHead"
            raise ValueError(msg)
        if init_cfg is None:
            init_cfg = {"type": "Normal", "layer": "Conv2d", "std": 0.01}

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            feat_channels=feat_channels,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            use_sigmoid_cls=use_sigmoid_cls,
        )

        self.nms_iou_threshold = nms_iou_threshold
        self.nms_pre = nms_pre
        self.with_nms = with_nms
        self.min_bbox_size = min_bbox_size
        self.max_per_img = max_per_img
        self.pos_weight = pos_weight
        self.reg_decoded_bbox = reg_decoded_bbox
        self.allowed_border = allowed_border
        self.score_threshold = score_threshold

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

    def prepare_loss_inputs(
        self,
        x: tuple[Tensor],
        entity: InstanceSegBatchDataEntity,  # type: ignore[override]
    ) -> tuple:
        """Perform forward propagation and prepare outputs for loss calculation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            entity (InstanceSegBatchDataEntity): Entity from OTX dataset.

        Returns:
            dict: A dictionary of components for loss calculation.
        """
        batch_gt_instances, batch_img_metas = unpack_inst_seg_entity(entity)
        cls_scores, bbox_preds = self(x)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
        )

        predictions = self.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_img_metas=batch_img_metas,
            cfg=self.test_cfg,
        )

        return cls_reg_targets, bbox_preds, cls_scores, predictions

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

        return self.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            cfg=self.test_cfg,
        )

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
        img_shape = img_meta["img_shape"]

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
            scores = cls_score.sigmoid()

            scores = torch.squeeze(scores)
            if 0 < self.nms_pre < scores.shape[0]:
                # sort is faster than topk
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[: self.nms_pre]
                scores = ranked_scores[: self.nms_pre]
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

        return self._bbox_post_process(results=results, rescale=rescale, img_meta=img_meta, cfg={})

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
            img_meta (dict, optional): Image meta info. Defaults to None.
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
        if not self.with_nms:
            msg = "`with_nms` must be True in RPNHead"
            raise RuntimeError(msg)

        if rescale:
            msg = "Rescale is not implemented in RPNHead"
            raise NotImplementedError

        # filter small size bboxes
        if self.min_bbox_size >= 0:
            w, h = get_box_wh(results.bboxes)  # type: ignore[attr-defined]
            valid_mask = (w > self.min_bbox_size) & (h > self.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:  # type: ignore[attr-defined]
            bboxes = results.bboxes  # type: ignore[attr-defined]
            nms_cfg = {
                "type": "nms",
                "iou_threshold": self.nms_iou_threshold,
            }  # TODO(Kirill, Sungchul): depricate
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.level_ids, nms_cfg)  # type: ignore[attr-defined]
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[: self.max_per_img]

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
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
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

        batch_size = mlvl_cls_scores[0].shape[0]
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
            cls_score = cls_score.reshape(batch_size, -1)
            scores = cls_score.sigmoid()

            scores = scores.reshape(batch_size, -1, 1)
            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, dim)
            anchors = anchors.unsqueeze(0)

            if self.nms_pre > 0:
                _, topk_inds = dynamic_topk(scores.squeeze(2), self.nms_pre)
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
        batch_mlvl_scores = batch_mlvl_scores[..., : self.num_classes]
        if not self.with_nms:
            return batch_mlvl_bboxes, batch_mlvl_scores

        pre_top_k = 5000

        # only one class in rpn
        max_output_boxes_per_class = self.max_per_img

        return multiclass_nms(
            batch_mlvl_bboxes,
            batch_mlvl_scores,
            max_output_boxes_per_class,
            iou_threshold=self.nms_iou_threshold,
            score_threshold=self.score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=self.max_per_img,
        )


class RPNHead:
    """RPNHead factory for instance segmentation regional proposal network."""

    RPNHEAD_CFG: ClassVar[dict[str, Any]] = {
        "maskrcnn_resnet_50": {
            "in_channels": 256,
            "feat_channels": 256,
        },
        "maskrcnn_efficientnet_b2b": {
            "in_channels": 80,
            "feat_channels": 80,
            "max_per_img": 500,
            "nms_iou_threshold": 0.8,
            "nms_pre": 800,
        },
        "maskrcnn_swin_tiny": {
            "in_channels": 256,
            "feat_channels": 256,
        },
    }

    def __new__(
        cls,
        model_name: str,
        anchor_generator: AnchorGenerator,
        bbox_coder: DeltaXYWHBBoxCoder,
        assigner: MaxIoUAssigner,
        sampler: BaseSampler,
        train_cfg: dict,
        test_cfg: dict,
    ) -> RPNHeadModule:
        """RPNHead factory for instance segmentation regional proposal network."""
        return RPNHeadModule(
            **cls.RPNHEAD_CFG[model_name],
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            assigner=assigner,
            sampler=sampler,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
