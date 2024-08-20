# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.dense_heads.rtmdet_ins_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/dense_heads/rtmdet_ins_head.py
"""

from __future__ import annotations

import copy
import math
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn.functional
from datumaro import Polygon
from torch import Tensor, nn

from otx.algo.common.utils.nms import batched_nms, multiclass_nms
from otx.algo.common.utils.utils import (
    distance2bbox,
    filter_scores_and_topk,
    inverse_sigmoid,
    multi_apply,
    reduce_mean,
    select_single_mlvl,
)
from otx.algo.detection.heads.rtmdet_head import RTMDetHead
from otx.algo.instance_segmentation.utils.roi_extractors import OTXRoIAlign
from otx.algo.instance_segmentation.utils.structures.bbox.transforms import get_box_wh, scale_boxes
from otx.algo.instance_segmentation.utils.utils import unpack_inst_seg_entity
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.modules.norm import build_norm_layer, is_norm
from otx.algo.utils.mmengine_utils import InstanceData
from otx.algo.utils.weight_init import bias_init_with_prob, constant_init, normal_init
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity
from otx.core.utils.mask_util import polygon_to_bitmap

from .utils import sigmoid_geometric_mean

# mypy: disable-error-code="call-overload, index, override, attr-defined, misc"


class RTMDetInsHead(RTMDetHead):
    """Detection Head of RTMDet-Ins.

    Args:
        loss_mask (nn.Module): A module for mask loss.
        num_prototypes (int): Number of mask prototype features extracted
            from the mask head. Defaults to 8.
        dyconv_channels (int): Channel of the dynamic conv layers.
            Defaults to 8.
        num_dyconvs (int): Number of the dynamic convolution layers.
            Defaults to 3.
        mask_loss_stride (int): Down sample stride of the masks for loss
            computation. Defaults to 4.
    """

    def __init__(
        self,
        *args,
        loss_mask: nn.Module,
        num_prototypes: int = 8,
        dyconv_channels: int = 8,
        num_dyconvs: int = 3,
        mask_loss_stride: int = 4,
        **kwargs,
    ) -> None:
        self.num_prototypes = num_prototypes
        self.num_dyconvs = num_dyconvs
        self.dyconv_channels = dyconv_channels
        self.mask_loss_stride = mask_loss_stride
        super().__init__(*args, **kwargs)
        self.loss_mask = loss_mask

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        # a branch to predict kernels of dynamic convs
        self.kernel_convs = nn.ModuleList()
        # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(
                    # mask prototype and coordinate features
                    (self.num_prototypes + 2) * self.dyconv_channels,
                )
                bias_nums.append(self.dyconv_channels * 1)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels * 1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.kernel_convs.append(
                Conv2dModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    normalization=build_norm_layer(self.normalization, num_features=self.feat_channels),
                    activation=build_activation_layer(self.activation),
                ),
            )
        pred_pad_size = self.pred_kernel_size // 2
        self.rtm_kernel = nn.Conv2d(
            self.feat_channels,
            self.num_gen_params,
            self.pred_kernel_size,
            padding=pred_pad_size,
        )
        self.mask_head = MaskFeatModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            stacked_convs=4,
            num_levels=len(self.prior_generator.strides),
            num_prototypes=self.num_prototypes,
            activation=self.activation,
            normalization=self.normalization,
        )

    def forward(self, feats: tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - kernel_preds (list[Tensor]): Dynamic conv kernels for all scale
              levels, each is a 4D-tensor, the channels number is
              num_gen_params.
            - mask_feat (Tensor): Output feature of the mask head. Each is a
              4D-tensor, the channels number is num_prototypes.
        """
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for x, scale, stride in zip(feats, self.scales, self.prior_generator.strides):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls(cls_feat)

            for kernel_layer in self.kernel_convs:
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = self.rtm_kernel(kernel_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj(reg_feat)
                cls_score = inverse_sigmoid(sigmoid_geometric_mean(cls_score, objectness))

            reg_dist = scale(self.rtm_reg(reg_feat)) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat

    def predict_by_feat(
        self,
        cls_scores: tuple[Tensor],
        bbox_preds: tuple[Tensor],
        kernel_preds: tuple[Tensor],
        mask_feat: Tensor,
        batch_img_metas: list[dict],
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> list[InstanceData]:
        """Transform a batch of output features extracted from the head into bbox results."""
        if len(cls_scores) != len(bbox_preds):
            msg = "The length of cls_scores and bbox_preds should be the same."
            raise ValueError(msg)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True,
        )

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
            kernel_pred_list = select_single_mlvl(kernel_preds, img_id, detach=True)
            score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                kernel_pred_list=kernel_pred_list,
                mask_feat=mask_feat[img_id],
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
        kernel_pred_list: list[Tensor],
        mask_feat: Tensor,
        score_factor_list: list[Tensor],
        mlvl_priors: list[Tensor],
        img_meta: dict,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceData:
        """Transform a single image's features extracted from the head into bbox and mask results."""
        test_cfg = self.test_cfg if cfg is None else cfg
        test_cfg = copy.deepcopy(test_cfg)
        img_shape = img_meta["img_shape"]
        nms_pre = test_cfg.get("nms_pre", -1)  # type: ignore[union-attr]

        mlvl_bbox_preds = []
        mlvl_kernels = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []

        for cls_score, bbox_pred, kernel_pred, priors in zip(
            cls_score_list,
            bbox_pred_list,
            kernel_pred_list,
            mlvl_priors,
        ):
            if cls_score.size()[-2:] != bbox_pred.size()[-2:]:
                msg = "The spatial sizes of cls_score and bbox_pred should be the same."
                raise ValueError(msg)

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)  # noqa: PLW2901
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)  # noqa: PLW2901
            kernel_pred = kernel_pred.permute(1, 2, 0).reshape(-1, self.num_gen_params)  # noqa: PLW2901
            scores = cls_score.sigmoid() if self.use_sigmoid_cls else cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = test_cfg.get("score_thr", 0)  # type: ignore[union-attr]

            (
                scores,
                labels,
                _,
                filtered_results,
            ) = filter_scores_and_topk(
                scores,
                score_thr,
                nms_pre,
                {"bbox_pred": bbox_pred, "priors": priors, "kernel_pred": kernel_pred},
            )

            bbox_pred = filtered_results["bbox_pred"]  # noqa: PLW2901
            priors = filtered_results["priors"]  # noqa: PLW2901
            kernel_pred = filtered_results["kernel_pred"]  # noqa: PLW2901

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_kernels.append(kernel_pred)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors[..., :2], bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.priors = priors
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.kernels = torch.cat(mlvl_kernels)

        return self._bbox_mask_post_process(
            results=results,
            mask_feat=mask_feat,
            cfg=test_cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta,
        )

    def mask_postprocess(
        self,
        mask_logits: Tensor,
        img_h: int,
        img_w: int,
        gpu_mem_limit: float = 1.0,
        threshold: float = 0.5,
    ) -> Tensor:
        """Postprocess mask logits to binary masks.

        Args:
            mask_logits (Tensor): Mask Logits with shape (B, N, H, W).
            img_h (int): Image height resizes to.
            img_w (int): Image width resizes to.
            gpu_mem_limit (float, optional): GPU memory limit in GB. Defaults to 1.0.
            threshold (float, optional): Threshold for binary masks. Defaults to 0.5.

        Returns:
            Tensor: Binary masks with shape (B, N, img_h, img_w).
        """
        masks = torch.zeros(
            size=(mask_logits.shape[:2] + (img_h, img_w)),
            dtype=torch.bool,
            device=mask_logits.device,
        )

        total_bytes = mask_logits.element_size() * masks.nelement()
        num_chunks = int(math.ceil(total_bytes / (gpu_mem_limit * 1024) ** 3))
        n = mask_logits.shape[1]
        chunks = torch.chunk(
            torch.arange(n, device=mask_logits.device),
            num_chunks,
        )
        mask_logits = mask_logits.sigmoid()
        for inds in chunks:
            masks[:, inds] = (
                torch.nn.functional.interpolate(
                    mask_logits[:, inds],
                    size=[
                        img_h,
                        img_w,
                    ],
                    mode="bilinear",
                    align_corners=False,
                )
                >= threshold
            ).to(dtype=torch.bool)
        return masks

    def _bbox_mask_post_process(
        self,
        results: InstanceData,
        mask_feat: Tensor,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
        img_meta: dict | None = None,
    ) -> InstanceData:
        """Bbox and mask post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (InstaceData): Detection instance results,
                each item has shape (num_bboxes, ).
            mask_feat (Tensor): Mask prototype features of a single image
            cfg (dict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

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
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        if img_meta is None:
            img_meta = {}
            img_meta["scale_factor"] = [1.0, 1.0]

        stride = self.prior_generator.strides[0][0]
        if rescale:
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, "score_factors"):
            score_factors = results.pop("score_factors")
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg["min_bbox_size"] >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg["min_bbox_size"]) & (h > cfg["min_bbox_size"])
            if not valid_mask.all():
                results = results[valid_mask]

        if not with_nms:
            msg = "with_nms must be True for RTMDet-Ins"
            raise RuntimeError(msg)

        if results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores, results.labels, cfg["nms"])
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[: cfg["max_per_img"]]

            # process masks
            mask_logits = self._mask_predict_by_feat_single(mask_feat, results.kernels, results.priors)
            mask_logits = torch.nn.functional.interpolate(
                mask_logits.unsqueeze(0),
                scale_factor=stride,
                mode="bilinear",
            )

            if rescale:
                ori_h, ori_w = img_meta["ori_shape"][:2]
                masks = self.mask_postprocess(
                    mask_logits,
                    math.ceil(mask_logits.shape[-1] * scale_factor[1]),
                    math.ceil(mask_logits.shape[-2] * scale_factor[0]),
                    threshold=cfg["mask_thr_binary"],
                )[..., :ori_h, :ori_w]
                masks = masks.squeeze(0)
            else:
                masks = mask_logits.sigmoid().squeeze(0)
                masks = masks > cfg["mask_thr_binary"]
            results.masks = masks
        else:
            h, w = img_meta["ori_shape"][:2] if rescale else img_meta["img_shape"][:2]
            results.masks = torch.zeros(
                size=(results.bboxes.shape[0], h, w),
                dtype=torch.bool,
                device=results.bboxes.device,
            )

        return results

    def parse_dynamic_params(self, flatten_kernels: Tensor) -> tuple:
        """Split kernel head prediction to conv weight and bias."""
        n_inst = flatten_kernels.size(0)
        n_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(flatten_kernels, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for i in range(n_layers):
            if i < n_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(n_inst * self.dyconv_channels, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst * self.dyconv_channels)
            else:
                weight_splits[i] = weight_splits[i].reshape(n_inst, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst)

        return weight_splits, bias_splits

    def _mask_predict_by_feat_single(self, mask_feat: Tensor, kernels: Tensor, priors: Tensor) -> Tensor:
        """Generate mask logits from mask features with dynamic convs.

        Args:
            mask_feat (Tensor): Mask prototype features.
                Has shape (num_prototypes, H, W).
            kernels (Tensor): Kernel parameters for each instance.
                Has shape (num_instance, num_params)
            priors (Tensor): Center priors for each instance.
                Has shape (num_instance, 4).

        Returns:
            Tensor: Instance segmentation masks for each instance.
                Has shape (num_instance, H, W).
        """
        num_inst = priors.shape[0]
        h, w = mask_feat.size()[-2:]
        if num_inst < 1:
            return torch.empty(size=(num_inst, h, w), dtype=mask_feat.dtype, device=mask_feat.device)
        if len(mask_feat.shape) < 4:
            mask_feat.unsqueeze(0)

        coord = self.prior_generator.single_level_grid_priors((h, w), level_idx=0, device=mask_feat.device).reshape(
            1,
            -1,
            2,
        )
        num_inst = priors.shape[0]
        points = priors[:, :2].reshape(-1, 1, 2)
        strides = priors[:, 2:].reshape(-1, 1, 2)
        relative_coord = (points - coord).permute(0, 2, 1) / (strides[..., 0].reshape(-1, 1, 1) * 8)
        relative_coord = relative_coord.reshape(num_inst, 2, h, w)

        mask_feat = torch.cat([relative_coord, mask_feat.repeat(num_inst, 1, 1, 1)], dim=1)
        weights, biases = self.parse_dynamic_params(kernels)

        n_layers = len(weights)
        x = mask_feat.reshape(1, -1, h, w)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            x = torch.nn.functional.conv2d(x, weight, bias=bias, stride=1, padding=0, groups=num_inst)
            if i < n_layers - 1:
                x = torch.nn.functional.relu(x)
        return x.reshape(num_inst, h, w)

    def loss_mask_by_feat(
        self,
        mask_feats: Tensor,
        flatten_kernels: Tensor,
        sampling_results_list: list,
        batch_gt_instances: list[InstanceData],
    ) -> Tensor:
        """Compute instance segmentation loss.

        Args:
            mask_feats (list[Tensor]): Mask prototype features extracted from
                the mask head. Has shape (N, num_prototypes, H, W)
            flatten_kernels (list[Tensor]): Kernels of the dynamic conv layers.
                Has shape (N, num_instances, num_params)
            sampling_results_list (list[SamplingResults]) Batch of
                assignment results.
            batch_gt_instances (list[InstanceData]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            Tensor: The mask loss tensor.
        """
        batch_pos_mask_logits = []
        pos_gt_masks = []
        for mask_feat, kernels, sampling_results, gt_instances in zip(
            mask_feats,
            flatten_kernels,
            sampling_results_list,
            batch_gt_instances,
        ):
            pos_priors = sampling_results.pos_priors
            pos_inds = sampling_results.pos_inds
            pos_kernels = kernels[pos_inds]  # n_pos, num_gen_params
            pos_mask_logits = self._mask_predict_by_feat_single(mask_feat, pos_kernels, pos_priors)
            if gt_instances.masks.numel() == 0:
                gt_masks = torch.empty_like(gt_instances.masks)
            else:
                gt_masks = gt_instances.masks[sampling_results.pos_assigned_gt_inds, :]
            batch_pos_mask_logits.append(pos_mask_logits)
            pos_gt_masks.append(gt_masks)

        pos_gt_masks = torch.cat(pos_gt_masks, 0)
        batch_pos_mask_logits = torch.cat(batch_pos_mask_logits, 0)

        # avg_factor
        num_pos = batch_pos_mask_logits.shape[0]
        num_pos = (
            reduce_mean(
                mask_feats.new_tensor(
                    [
                        num_pos,
                    ],
                ),
            )
            .clamp_(min=1)
            .item()
        )

        if batch_pos_mask_logits.shape[0] == 0:
            return mask_feats.sum() * 0

        scale = self.prior_generator.strides[0][0] // self.mask_loss_stride
        # upsample pred masks
        batch_pos_mask_logits = torch.nn.functional.interpolate(
            batch_pos_mask_logits.unsqueeze(0),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        # downsample gt masks
        pos_gt_masks = pos_gt_masks[
            :,
            self.mask_loss_stride // 2 :: self.mask_loss_stride,
            self.mask_loss_stride // 2 :: self.mask_loss_stride,
        ]

        return self.loss_mask(batch_pos_mask_logits, pos_gt_masks, weight=None, avg_factor=num_pos)

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        kernel_preds: list[Tensor],
        mask_feat: Tensor,
        batch_gt_instances: list[InstanceData],
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | None = None,
    ) -> dict[str, Tensor]:
        """Compute losses of the head."""
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        if len(featmap_sizes) != self.prior_generator.num_levels:
            msg = "The number of featmap sizes should be equal to the number of levels."
            raise ValueError(msg)

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat(
            [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in cls_scores],
            1,
        )
        flatten_kernels = torch.cat(
            [
                kernel_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_gen_params)
                for kernel_pred in kernel_preds
            ],
            1,
        )
        decoded_bboxes = []
        for anchor, bbox_pred in zip(anchor_list[0], bbox_preds):
            anchor = anchor.reshape(-1, 4)  # noqa: PLW2901
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)  # noqa: PLW2901
            bbox_pred = distance2bbox(anchor, bbox_pred)  # noqa: PLW2901
            decoded_bboxes.append(bbox_pred)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)
        # Convert polygon masks to bitmap masks
        if isinstance(batch_gt_instances[0].masks[0], Polygon):
            for gt_instances, img_meta in zip(batch_gt_instances, batch_img_metas):
                ndarray_masks = polygon_to_bitmap(gt_instances.masks, *img_meta["img_shape"])
                if len(ndarray_masks) == 0:
                    ndarray_masks = np.empty((0, *img_meta["img_shape"]), dtype=np.uint8)
                gt_instances.masks = torch.tensor(ndarray_masks, dtype=torch.bool, device=device)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )
        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            sampling_results_list,
        ) = cls_reg_targets

        losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            decoded_bboxes,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            self.prior_generator.strides,
        )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = [x / cls_avg_factor for x in losses_cls]

        bbox_avg_factor = reduce_mean(sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = [x / bbox_avg_factor for x in losses_bbox]

        loss_mask = self.loss_mask_by_feat(mask_feat, flatten_kernels, sampling_results_list, batch_gt_instances)
        return {"loss_cls": losses_cls, "loss_bbox": losses_bbox, "loss_mask": loss_mask}


class MaskFeatModule(BaseModule):
    """Mask feature head used in RTMDet-Ins.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
            map branch.
        num_levels (int): The starting feature map level from RPN that
            will be used to predict the mask feature map.
        num_prototypes (int): Number of output channel of the mask feature
            map branch. This is the channel count of the mask
            feature map that to be dynamically convolved with the predicted
            kernel.
        stacked_convs (int): Number of convs in mask feature branch.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``partial(nn.ReLU, inplace=True)``.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] | int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        num_levels: int = 3,
        num_prototypes: int = 8,
        activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
        normalization: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__(init_cfg=None)

        self.num_levels = num_levels
        self.fusion_conv = nn.Conv2d(num_levels * in_channels, in_channels, 1)
        convs = []
        for i in range(stacked_convs):
            in_c = in_channels if i == 0 else feat_channels
            convs.append(
                Conv2dModule(
                    in_c,
                    feat_channels,
                    3,
                    padding=1,
                    normalization=build_norm_layer(normalization, num_features=feat_channels),
                    activation=build_activation_layer(activation),
                ),
            )
        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(feat_channels, num_prototypes, kernel_size=1)

    def forward(self, features: tuple[Tensor, ...]) -> Tensor:
        """Forward features from the upstream network."""
        # multi-level feature fusion
        fusion_feats = [features[0]]
        size = features[0].shape[-2:]
        for i in range(1, self.num_levels):
            f = torch.nn.functional.interpolate(features[i], size=size, mode="bilinear")
            fusion_feats.append(f)
        fusion_feats = torch.cat(fusion_feats, dim=1)
        fusion_feats = self.fusion_conv(fusion_feats)
        # pred mask feats
        mask_features = self.stacked_convs(fusion_feats)
        return self.projection(mask_features)


class RTMDetInsSepBNHead(RTMDetInsHead):
    """Detection Head of RTMDet-Ins with sep-bn layers.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, requires_grad=True)``.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``partial(nn.SiLU, inplace=True)``.
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        share_conv: bool = True,
        with_objectness: bool = False,
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, requires_grad=True),
        activation: Callable[..., nn.Module] = partial(nn.SiLU, inplace=True),
        pred_kernel_size: int = 1,
        **kwargs,
    ) -> None:
        self.share_conv = share_conv
        super().__init__(
            num_classes,
            in_channels,
            normalization=normalization,
            activation=activation,
            pred_kernel_size=pred_kernel_size,
            with_objectness=with_objectness,
            **kwargs,
        )

        self.roi_align = OTXRoIAlign(
            output_size=(28, 28),
            sampling_ratio=0,
            aligned=True,
            spatial_scale=1.0,
        )

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_kernel = nn.ModuleList()
        self.rtm_obj = nn.ModuleList()

        # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append((self.num_prototypes + 2) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        pred_pad_size = self.pred_kernel_size // 2

        for _ in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            kernel_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    Conv2dModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        normalization=build_norm_layer(self.normalization, num_features=self.feat_channels),
                        activation=build_activation_layer(self.activation),
                    ),
                )
                reg_convs.append(
                    Conv2dModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        normalization=build_norm_layer(self.normalization, num_features=self.feat_channels),
                        activation=build_activation_layer(self.activation),
                    ),
                )
                kernel_convs.append(
                    Conv2dModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        normalization=build_norm_layer(self.normalization, num_features=self.feat_channels),
                        activation=build_activation_layer(self.activation),
                    ),
                )
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(cls_convs)
            self.kernel_convs.append(kernel_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=pred_pad_size,
                ),
            )
            self.rtm_reg.append(
                nn.Conv2d(self.feat_channels, self.num_base_priors * 4, self.pred_kernel_size, padding=pred_pad_size),
            )
            self.rtm_kernel.append(
                nn.Conv2d(self.feat_channels, self.num_gen_params, self.pred_kernel_size, padding=pred_pad_size),
            )
            if self.with_objectness:
                self.rtm_obj.append(nn.Conv2d(self.feat_channels, 1, self.pred_kernel_size, padding=pred_pad_size))

        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

        self.mask_head = MaskFeatModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            stacked_convs=4,
            num_levels=len(self.prior_generator.strides),
            num_prototypes=self.num_prototypes,
            activation=self.activation,
            normalization=self.normalization,
        )

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg in zip(self.rtm_cls, self.rtm_reg):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01, bias=1)
        if self.with_objectness:
            for rtm_obj in self.rtm_obj:
                normal_init(rtm_obj, std=0.01, bias=bias_cls)

    def forward(self, feats: tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - kernel_preds (list[Tensor]): Dynamic conv kernels for all scale
              levels, each is a 4D-tensor, the channels number is
              num_gen_params.
            - mask_feat (Tensor): Output feature of the mask head. Each is a
              4D-tensor, the channels number is num_prototypes.
        """
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, (x, stride) in enumerate(zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for kernel_layer in self.kernel_convs[idx]:
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = self.rtm_kernel[idx](kernel_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(sigmoid_geometric_mean(cls_score, objectness))

            reg_dist = torch.nn.functional.relu(self.rtm_reg[idx](reg_feat)) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat

    def loss(self, x: tuple[Tensor], entity: InstanceSegBatchDataEntity) -> dict:
        """Perform forward propagation and loss calculation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            entity (InstanceSegBatchDataEntity): Entity from OTX dataset.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        batch_gt_instances, batch_img_metas = unpack_inst_seg_entity(entity)

        loss_inputs = (*outs, batch_gt_instances, batch_img_metas)
        return self.loss_by_feat(*loss_inputs)

    def export_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        kernel_preds: list[Tensor],
        mask_feat: Tensor,
        batch_img_metas: list[dict] | None = None,
        cfg: dict | None = None,
        rescale: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Export the detection head."""
        if len(cls_scores) != len(bbox_preds):
            msg = "The length of cls_scores and bbox_preds should be the same."
            raise ValueError(msg)

        device = cls_scores[0].device
        cfg = self.test_cfg if cfg is None else cfg
        batch_size = bbox_preds[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, device=device, with_stride=True)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for bbox_pred in bbox_preds]
        flatten_kernel_preds = [
            kernel_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_gen_params) for kernel_pred in kernel_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        _flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_kernel_preds = torch.cat(flatten_kernel_preds, dim=1)
        priors = torch.cat(mlvl_priors)
        tl_x = priors[..., 0] - _flatten_bbox_preds[..., 0]
        tl_y = priors[..., 1] - _flatten_bbox_preds[..., 1]
        br_x = priors[..., 0] + _flatten_bbox_preds[..., 2]
        br_y = priors[..., 1] + _flatten_bbox_preds[..., 3]
        bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        scores = flatten_cls_scores

        max_output_boxes_per_class = 100
        iou_threshold = cfg["nms"].get("iou_threshold", 0.5)  # type: ignore[union-attr]
        score_threshold = cfg.get("score_thr", 0.05)  # type: ignore[union-attr]
        pre_top_k = 300
        keep_top_k = cfg.get("max_per_img", 100)  # type: ignore[union-attr]

        return self._nms_with_mask_static(
            priors,
            bboxes,
            scores,
            flatten_kernel_preds,
            mask_feat,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            pre_top_k,
            keep_top_k,
        )

    def _nms_with_mask_static(
        self,
        priors: Tensor,
        bboxes: Tensor,
        scores: Tensor,
        kernels: Tensor,
        mask_feats: Tensor,
        max_output_boxes_per_class: int = 1000,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.05,
        pre_top_k: int = -1,
        keep_top_k: int = -1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Wrapper for `multiclass_nms` with ONNXRuntime.

        Note:
            Compared with the original _nms_with_mask_static, this function
            crops masks using RoIAlign and returns the cropped masks.

        Args:
            self: The instance of `RTMDetInsHead`.
            priors (Tensor): The prior boxes of shape [num_boxes, 4].
            bboxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            kernels (Tensor): The dynamic conv kernels.
            mask_feats (Tensor): The mask feature.
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms. Defaults to 1000.
            iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
            score_threshold (float): score threshold of nms.
                Defaults to 0.05.
            pre_top_k (int): Number of top K boxes to keep before nms.
                Defaults to -1.
            keep_top_k (int): Number of top K boxes to keep after nms.
                Defaults to -1.
            mask_thr_binary (float): Binarization threshold for masks.

        Returns:
            tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
                and `labels` of shape [N, num_det].
        """
        dets, labels, inds = multiclass_nms(
            bboxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k,
            output_index=True,
        )

        batch_size = bboxes.shape[0]
        batch_inds = torch.arange(batch_size, device=bboxes.device).view(-1, 1)
        kernels = kernels[batch_inds, inds, :]
        priors = priors.unsqueeze(0).repeat(batch_size, 1, 1)
        priors = priors[batch_inds, inds, :]
        mask_logits = self.export_mask_predict_by_feat_single(mask_feats, kernels, priors)
        stride = self.prior_generator.strides[0][0]
        mask_logits = torch.nn.functional.interpolate(mask_logits, scale_factor=stride, mode="bilinear")
        masks = mask_logits.sigmoid()

        batch_index = (
            torch.arange(dets.size(0), device=dets.device).float().view(-1, 1, 1).expand(dets.size(0), dets.size(1), 1)
        )
        rois = torch.cat([batch_index, dets[..., :4]], dim=-1)
        cropped_masks = self.roi_align.export(masks, rois[0])
        cropped_masks = cropped_masks[torch.arange(cropped_masks.size(0)), torch.arange(cropped_masks.size(0))]
        cropped_masks = cropped_masks.unsqueeze(0)
        return dets, labels, cropped_masks

    def export_mask_predict_by_feat_single(
        self,
        mask_feat: Tensor,
        kernels: Tensor,
        priors: Tensor,
    ) -> Tensor:
        """Decode mask with dynamic conv.

        Note: Prior Generator has cuda device set as default.
        However, this would cause some problems on CPU only devices.
        """
        num_inst = priors.shape[1]
        batch_size = priors.shape[0]
        hw = mask_feat.size()[-2:]
        # NOTE: had to force to set device in prior generator
        coord = self.prior_generator.single_level_grid_priors(hw, level_idx=0, device=mask_feat.device).to(
            mask_feat.device,
        )
        coord = coord.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        priors = priors.unsqueeze(2)
        points = priors[..., :2]
        relative_coord = (points - coord).permute(0, 1, 3, 2) / (priors[..., 2:3] * 8)
        relative_coord = relative_coord.reshape(batch_size, num_inst, 2, hw[0], hw[1])

        mask_feat = torch.cat([relative_coord, mask_feat.unsqueeze(1).repeat(1, num_inst, 1, 1, 1)], dim=2)
        weights, biases = self._parse_dynamic_params(kernels)

        n_layers = len(weights)
        x = mask_feat.flatten(0, 1).flatten(2)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            # replace dynamic conv with bmm
            weight = weight.flatten(0, 1)  # noqa: PLW2901
            bias = bias.flatten(0, 1).unsqueeze(2)  # noqa: PLW2901
            x = torch.bmm(weight, x)
            x = x + bias
            if i < n_layers - 1:
                x = x.clamp_(min=0)
        return x.reshape(batch_size, num_inst, hw[0], hw[1])

    def _parse_dynamic_params(self, flatten_kernels: Tensor) -> tuple:
        """Split kernel head prediction to conv weight and bias."""
        batch_size = flatten_kernels.shape[0]
        n_inst = flatten_kernels.shape[1]
        n_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(flatten_kernels, self.weight_nums + self.bias_nums, dim=2))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for idx in range(n_layers):
            channel = self.dyconv_channels if idx < n_layers - 1 else 1
            weight_splits[idx] = weight_splits[idx].reshape(batch_size, n_inst, channel, -1)

        return weight_splits, bias_splits
