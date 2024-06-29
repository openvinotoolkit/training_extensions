# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.dense_heads.base_dense_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/dense_heads/base_dense_head.py
"""

from __future__ import annotations

import copy
from abc import abstractmethod

import torch
from torch import Tensor

from otx.algo.common.utils.nms import batched_nms, multiclass_nms
from otx.algo.common.utils.utils import dynamic_topk, filter_scores_and_topk, gather_topk, select_single_mlvl
from otx.algo.detection.utils.utils import unpack_det_entity
from otx.algo.modules.base_module import BaseModule
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import OTXBatchDataEntity
from otx.core.data.entity.detection import DetBatchDataEntity


class BaseDenseHead(BaseModule):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg: dict | list[dict] | None = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # `_raw_positive_infos` will be used in `get_positive_infos`, which
        # can get positive information.
        self._raw_positive_infos: dict = {}

    def get_positive_infos(self) -> list[InstanceData] | None:
        """Get positive information from sampling results.

        Returns:
            list[InstanceData]: Positive information of each image,
            usually including positive bboxes, positive labels, positive
            priors, etc.
        """
        if len(self._raw_positive_infos) == 0:
            return None

        sampling_results = self._raw_positive_infos.get("sampling_results", None)
        positive_infos = []
        for sampling_result in sampling_results:
            pos_info = InstanceData()
            pos_info.bboxes = sampling_result.pos_gt_bboxes
            pos_info.labels = sampling_result.pos_gt_labels
            pos_info.priors = sampling_result.pos_priors
            pos_info.pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            pos_info.pos_inds = sampling_result.pos_inds
            positive_infos.append(pos_info)
        return positive_infos

    def loss(self, x: tuple[Tensor], entity: DetBatchDataEntity) -> dict:
        """Perform forward propagation and loss calculation of the detection head.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            entity (DetBatchDataEntity): Entity from OTX dataset.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        batch_gt_instances, batch_img_metas = unpack_det_entity(entity)

        loss_inputs = (*outs, batch_gt_instances, batch_img_metas)
        return self.loss_by_feat(*loss_inputs)

    @abstractmethod
    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_gt_instances: list[InstanceData],
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | None = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection head."""

    def predict(
        self,
        x: tuple[Tensor],
        entity: OTXBatchDataEntity,
        rescale: bool = False,
    ) -> list[InstanceData]:
        """Perform forward propagation of the detection head and predict detection results.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            entity (DetBatchDataEntity): Entity from OTX dataset.
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

        outs = self(x)

        return self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)  # type: ignore[misc]

    def predict_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        score_factors: list[Tensor] | None = None,
        batch_img_metas: list[dict] | None = None,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> list[InstanceData]:
        """Transform a batch of output features extracted from the head into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
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
            cfg (dict, optional): Test / postprocessing configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[InstanceData]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if batch_img_metas is None:
            batch_img_metas = []

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
        )

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
            if score_factors is not None:
                score_factor_list = select_single_mlvl(score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms,
            )
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(
        self,
        cls_score_list: list[Tensor],
        bbox_pred_list: list[Tensor],
        score_factor_list: list[Tensor],
        mlvl_priors: list[Tensor],
        img_meta: dict,
        cfg: dict | None = None,
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
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
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
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            InstanceData: Detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        with_score_factors = score_factor_list[0] is not None

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta["img_shape"]
        nms_pre = cfg.get("nms_pre", -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_score_factors: list | None = [] if with_score_factors else None
        for cls_score, bbox_pred, score_factor, priors in zip(
            cls_score_list,
            bbox_pred_list,
            score_factor_list,
            mlvl_priors,
        ):
            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)  # noqa: PLW2901
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()  # noqa: PLW2901
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)  # noqa: PLW2901

            # the `custom_cls_channels` parameter is derived from
            # CrossEntropyCustomLoss and FocalCustomLoss, and is currently used
            # in v3det.
            if getattr(self.loss_cls, "custom_cls_channels", False):
                scores = self.loss_cls.get_activation(cls_score)
            elif self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get("score_thr", 0)

            filtered_results: dict
            scores, labels, keep_idxs, filtered_results = filter_scores_and_topk(  # type: ignore[assignment]
                scores,
                score_thr,
                nms_pre,
                {"bbox_pred": bbox_pred, "priors": priors},
            )

            bbox_pred = filtered_results["bbox_pred"]  # noqa: PLW2901
            priors = filtered_results["priors"]  # noqa: PLW2901

            if with_score_factors:
                score_factor = score_factor[keep_idxs]  # noqa: PLW2901

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if mlvl_score_factors is not None:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(results=results, cfg=cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta)

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
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (InstaceData): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (dict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            InstanceData: Detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if rescale:
            scale_factor = [1 / s for s in img_meta["scale_factor"]]  # (H, W)
            results.bboxes = results.bboxes * results.bboxes.new_tensor(scale_factor[::-1]).repeat(  # type: ignore[attr-defined]
                (1, int(results.bboxes.size(-1) / 2)),  # type: ignore[attr-defined]
            )

        if hasattr(results, "score_factors"):
            score_factors = results.pop("score_factors")
            results.scores = results.scores * score_factors  # type: ignore[attr-defined]

        # filter small size bboxes
        if (min_bbox_size := cfg.get("min_bbox_size", -1)) >= 0:
            w = results.bboxes[:, 2] - results.bboxes[:, 0]  # type: ignore[attr-defined]
            h = results.bboxes[:, 3] - results.bboxes[:, 1]  # type: ignore[attr-defined]
            valid_mask = (w > min_bbox_size) & (h > min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if with_nms and results.bboxes.numel() > 0:  # type: ignore[attr-defined]
            bboxes = results.bboxes  # type: ignore[attr-defined]
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels, cfg["nms"])  # type: ignore[attr-defined]
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[: cfg["max_per_img"]]

        return results

    def export(
        self,
        x: tuple[Tensor],
        batch_img_metas: list[dict],
        rescale: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Perform forward propagation of the detection head and predict detection results.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream network, each is a 4D-tensor.
            batch_data_samples (list[dict]): The Data Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]]:
                Detection results of each image after the post process.
        """
        outs = self(x)

        return self.export_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)  # type: ignore[misc]

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
        """Transform a batch of output features extracted from the head into bbox results.

        Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/models/dense_heads/base_dense_head.py#L26-L206

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
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
            cfg (dict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if batch_img_metas is None:
            batch_img_metas = [{}]

        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
        )

        mlvl_priors = [priors.unsqueeze(0) for priors in mlvl_priors]

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        if score_factors is None:
            with_score_factors = False
            mlvl_score_factor = [None for _ in range(num_levels)]
        else:
            with_score_factors = True
            mlvl_score_factor = [score_factors[i].detach() for i in range(num_levels)]
            mlvl_score_factors = []

        img_shape = batch_img_metas[0]["img_shape"]
        batch_size = cls_scores[0].shape[0]
        cfg = cfg or self.test_cfg
        pre_topk = cfg.get("nms_pre", -1)

        mlvl_valid_bboxes = []
        mlvl_valid_scores = []
        mlvl_valid_priors = []

        for cls_score, bbox_pred, score_factors, priors in zip(
            mlvl_cls_scores,
            mlvl_bbox_preds,
            mlvl_score_factor,
            mlvl_priors,
        ):
            scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels)

            if getattr(self.loss_cls, "custom_cls_channels", False):
                scores = self.loss_cls.get_activation(cls_score)
            elif self.use_sigmoid_cls:
                scores = scores.sigmoid()
            else:
                scores = scores.softmax(-1)[:, :, :-1]

            if with_score_factors:
                score_factors = score_factors.permute(0, 2, 3, 1).reshape(batch_size, -1).sigmoid()  # type: ignore[union-attr, attr-defined] # noqa: PLW2901
                score_factors = score_factors.unsqueeze(2)  # type: ignore[union-attr] # noqa: PLW2901

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, dim)  # noqa: PLW2901

            if pre_topk > 0:
                nms_pre_score = scores
                if with_score_factors:
                    nms_pre_score = nms_pre_score * score_factors

                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = nms_pre_score.max(-1)
                else:
                    max_scores, _ = nms_pre_score[..., :-1].max(-1)
                _, topk_inds = dynamic_topk(max_scores, pre_topk)
                bbox_pred, scores, score_factors = gather_topk(  # noqa: PLW2901
                    bbox_pred,
                    scores,
                    score_factors,  # type: ignore[arg-type]
                    inds=topk_inds,
                    batch_size=batch_size,
                    is_batched=True,
                )
                priors = gather_topk(priors, inds=topk_inds, batch_size=batch_size, is_batched=False)  # noqa: PLW2901

            mlvl_valid_bboxes.append(bbox_pred)
            mlvl_valid_scores.append(scores)
            mlvl_valid_priors.append(priors)
            if with_score_factors:
                mlvl_score_factors.append(score_factors)

        batch_mlvl_bboxes_pred = torch.cat(mlvl_valid_bboxes, dim=1)
        batch_scores = torch.cat(mlvl_valid_scores, dim=1)
        batch_priors = torch.cat(mlvl_valid_priors, dim=1)

        batch_bboxes = self.bbox_coder.decode_export(batch_priors, batch_mlvl_bboxes_pred, max_shape=img_shape)

        if with_score_factors:
            batch_score_factors = torch.cat(mlvl_score_factors, dim=1)

        if not self.use_sigmoid_cls:
            batch_scores = batch_scores[..., : self.num_classes]

        if with_score_factors:
            batch_scores = batch_scores * batch_score_factors

        return multiclass_nms(
            batch_bboxes,
            batch_scores,
            max_output_boxes_per_class=200,  # TODO (sungchul): temporarily set to mmdeploy cfg, will be updated
            iou_threshold=cfg["nms"]["iou_threshold"],
            score_threshold=cfg["score_thr"],
            pre_top_k=5000,
            keep_top_k=cfg["max_per_img"],
        )
