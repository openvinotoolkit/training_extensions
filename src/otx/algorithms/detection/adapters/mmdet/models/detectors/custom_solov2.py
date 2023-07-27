"""OTX SOLOv2 Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List

import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.solov2 import SOLOv2

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.common.utils.logger import get_logger

from .l2sp_detector_mixin import L2SPDetectorMixin
from .sam_detector_mixin import SAMDetectorMixin

logger = get_logger()


@DETECTORS.register_module()
class CustomSOLOv2(SAMDetectorMixin, L2SPDetectorMixin, SOLOv2):
    """CustomSOLOv2 Class for mmdetection detectors."""

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)


if is_mmdeploy_enabled():
    import torch.nn.functional as F
    from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
    from mmdeploy.core import FUNCTION_REWRITER, mark
    from mmdeploy.utils import is_dynamic_shape
    from torch import Tensor

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_solov2.CustomSOLOv2.simple_test"
    )
    def custom_solov2__simple_test(ctx, self, img, img_metas, **kwargs):
        """Function for custom_solov2__simple_test."""
        feat = self.extract_feat(img)
        out = self.mask_head.simple_test(feat, img_metas, rescale=False)

        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(feat)
            # Saliency map will be generated from predictions. Generate dummy saliency_map.
            saliency_map = torch.empty(1, dtype=torch.uint8)
            return (*out, feature_vector, saliency_map)

        return out

    @mark("custom_solov2_forward", inputs=["input"], outputs=["boxes", "labels", "masks", "feats", "saliencies"])
    def __forward_impl(ctx, self, img, img_metas, **kwargs):
        """Internal Function for __forward_impl."""
        assert isinstance(img, torch.Tensor)

        deploy_cfg = ctx.cfg
        is_dynamic_flag = is_dynamic_shape(deploy_cfg)
        # get origin input shape as tensor to support onnx dynamic shape
        img_shape = torch._shape_as_tensor(img)[2:]
        if not is_dynamic_flag:
            img_shape = [int(val) for val in img_shape]
        img_metas[0]["img_shape"] = img_shape
        return self.simple_test(img, img_metas, **kwargs)

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_solov2.CustomSOLOv2.forward"
    )
    def custom_solov2__forward(ctx, self, img, img_metas=None, return_loss=False, **kwargs):
        """Internal Function for __forward for CustomSOLOv2."""
        if img_metas is None:
            img_metas = [{}]
        else:
            assert len(img_metas) == 1, "do not support aug_test"
            img_metas = img_metas[0]

        if isinstance(img, list):
            img = img[0]

        return __forward_impl(ctx, self, img, img_metas=img_metas, **kwargs)

    def mask_matrix_nms_onnx(
        masks, labels, scores, filter_thr=-1, nms_pre=-1, max_num=-1, kernel="gaussian", sigma=2.0, mask_area=None
    ):
        """Matrix NMS for multi-class masks.

        Args:
            masks (Tensor): Has shape (num_instances, h, w)
            labels (Tensor): Labels of corresponding masks,
                has shape (num_instances,).
            scores (Tensor): Mask scores of corresponding masks,
                has shape (num_instances).
            filter_thr (float): Score threshold to filter the masks
                after matrix nms. Default: -1, which means do not
                use filter_thr.
            nms_pre (int): The max number of instances to do the matrix nms.
                Default: -1, which means do not use nms_pre.
            max_num (int, optional): If there are more than max_num masks after
                matrix, only top max_num will be kept. Default: -1, which means
                do not use max_num.
            kernel (str): 'linear' or 'gaussian'.
            sigma (float): std in gaussian method.
            mask_area (Tensor): The sum of seg_masks.

        Returns:
            tuple(Tensor): Processed mask results.

                - scores (Tensor): Updated scores, has shape (n,).
                - labels (Tensor): Remained labels, has shape (n,).
                - masks (Tensor): Remained masks, has shape (n, w, h).
                - keep_inds (Tensor): The indices number of
                    the remaining mask in the input mask, has shape (n,).
        """
        assert len(labels) == len(masks) == len(scores)
        assert len(masks) == len(mask_area)
        # sort and keep top nms_pre
        nms_pre = max(0, nms_pre)
        if nms_pre <= 0:
            nms_pre = scores.shape[0]
        scores, sort_inds = torch.topk(scores, nms_pre)

        keep_inds = sort_inds
        masks = masks[sort_inds]
        mask_area = mask_area[sort_inds]
        labels = labels[sort_inds]
        num_masks = labels.size(0)
        flatten_masks = masks.reshape(num_masks, -1).float()
        # inter.
        inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
        expanded_mask_area = mask_area.expand(num_masks, num_masks)

        total_area = expanded_mask_area + expanded_mask_area.transpose(1, 0) - inter_matrix
        # Upper triangle iou matrix.
        total_area_mask = total_area > 0
        total_area = total_area.where(total_area_mask, total_area.new_ones(1))

        iou_matrix = (inter_matrix / total_area).triu(diagonal=1)
        # label_specific matrix.
        expanded_labels = labels.expand(num_masks, num_masks)
        # Upper triangle label matrix.
        label_matrix = (expanded_labels == expanded_labels.transpose(1, 0)).triu(diagonal=1)

        # IoU compensation
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)

        # IoU decay
        decay_iou = iou_matrix * label_matrix

        # calculate the decay_coefficient
        if kernel == "gaussian":
            decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
            compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
            decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
        elif kernel == "linear":
            decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
            decay_coefficient, _ = decay_matrix.min(0)
        else:
            raise NotImplementedError(f"{kernel} kernel is not supported in matrix nms!")
        # update the score.
        scores = scores * decay_coefficient

        keep = scores >= filter_thr
        scores = scores.where(keep, scores.new_zeros(1))

        # sort and keep top max_num
        scores, sort_inds = torch.topk(scores, max(max_num, 0))
        keep_inds = keep_inds[sort_inds]
        masks = masks[sort_inds]
        labels = labels[sort_inds]

        return scores, labels, masks, keep_inds

    @FUNCTION_REWRITER.register_rewriter(func_name="mmdet.models.dense_heads.solov2_head.SOLOV2Head.get_results")
    def solov2_head__get_results(
        ctx,
        self,
        mlvl_kernel_preds: List[Tensor],
        mlvl_cls_scores: List[Tensor],
        mask_feats: Tensor,
        batch_img_metas: List[Dict],
        **kwargs,
    ):
        """Rewrite `get_results` of `SOLOV2Head` for default backend.

        Args:
            ctx (Context): Context object.
            self (SOLOV2Head): SOLOV2Head instance.
            mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
                prediction. The kernel is used to generate instance
                segmentation masks by dynamic convolution. Each element in the
                list has shape
                (batch_size, kernel_out_channels, num_grids, num_grids).
            mlvl_cls_scores (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids, num_grids).
            mask_feats (Tensor): Unified mask feature map used to generate
                instance segmentation masks by dynamic convolution. Has shape
                (batch_size, mask_out_channels, h, w).
            batch_img_metas (list[dict]): Meta information of all images.
            kwargs (dict): Other arguments.

        Returns:
                - dets (Tensor): bboxes with scores, has shape (1, num_instance, 5).
                - labels (Tensor): Has shape (1, num_instances,).
                - masks (Tensor): Processed mask results, has
                    shape (1, num_instances, h, w).
        """

        cfg = self.test_cfg
        num_levels = len(mlvl_cls_scores)
        batch_size = mlvl_cls_scores[0].size(0)
        assert batch_size == 1, "batch size must be 1 for onnx export"
        assert len(mlvl_kernel_preds) == len(mlvl_cls_scores)

        for lvl in range(num_levels):
            kernel_preds = mlvl_kernel_preds[lvl]
            cls_scores = mlvl_cls_scores[lvl]
            cls_scores = cls_scores.sigmoid()
            local_max = F.max_pool2d(cls_scores, 2, stride=1, padding=1)
            keep_mask = local_max[:, :, :-1, :-1] == cls_scores
            cls_scores = cls_scores * keep_mask
            mlvl_cls_scores[lvl] = cls_scores.permute(0, 2, 3, 1).view(batch_size, -1, self.cls_out_channels)
            mlvl_kernel_preds[lvl] = kernel_preds.permute(0, 2, 3, 1).view(batch_size, -1, self.kernel_out_channels)

        # Rewrite strides to avoid set_items.
        mlvl_strides = [
            torch.ones_like(mlvl_cls_scores[lvl][0, :, 0]) * self.strides[lvl] for lvl in range(len(mlvl_cls_scores))
        ]
        strides = torch.cat(mlvl_strides, 0)
        assert len(mlvl_kernel_preds) == len(mlvl_cls_scores)
        batch_mlvl_cls_scores = torch.cat(mlvl_cls_scores, dim=1)
        batch_mlvl_kernel_preds = torch.cat(mlvl_kernel_preds, dim=1)

        featmap_size = mask_feats.size()[-2:]
        h, w = batch_img_metas[0]["img_shape"][:2]
        cls_scores = batch_mlvl_cls_scores[0]
        kernel_preds = batch_mlvl_kernel_preds[0]

        cls_scores, cls_labels = torch.max(cls_scores, -1)

        # NOTE:
        # When cfg.score_thr is set too high and random value input is used for tracing,
        # the score mask might end up empty, rendering the outputs untraceable.
        # To circumvent this issue, an alternative approach is to employ a real input image
        # during the tracing process. This ensures that the score mask contains meaningful data,
        # making the outputs traceable and facilitating the ONNX export.

        # score_mask = (cls_scores > cfg.score_thr)
        score_mask = cls_scores > 0.05
        cls_scores = cls_scores.where(score_mask, cls_scores.new_zeros(1))

        # mask encoding.
        # NOTE: ONNX does not support the following dynamic convolution:
        # kernel_preds = kernel_preds.unsqueeze(2).unsqueeze(3)
        # mask_preds = F.conv2d(mask_feats, kernel_preds, stride=1).squeeze(0).sigmoid()

        # NOTE: rewrite it with:
        kernel_preds = kernel_preds.view(kernel_preds.size(0), -1, self.dynamic_conv_size, self.dynamic_conv_size)
        unfolded_mask_feats = torch.nn.functional.unfold(mask_feats, (self.dynamic_conv_size, self.dynamic_conv_size))
        unfolded_mask_preds = (
            unfolded_mask_feats.transpose(1, 2).matmul(kernel_preds.view(kernel_preds.size(0), -1).t()).transpose(1, 2)
        )
        mask_preds = unfolded_mask_preds.view(
            unfolded_mask_preds.size(0), unfolded_mask_preds.size(1), featmap_size[0], featmap_size[1]
        )
        mask_preds = mask_preds.squeeze(0).sigmoid()

        aligned_score_mask = score_mask.unsqueeze(1).unsqueeze(2)
        mask_preds = mask_preds.where(aligned_score_mask, mask_preds.new_zeros(1))

        # mask.
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2))
        keep = sum_masks > strides
        cls_scores = cls_scores.where(keep, cls_scores.new_zeros(1))
        sum_masks = sum_masks.where(keep, sum_masks.new_ones(1))

        # maskness.
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores
        sum_masks = sum_masks.where(keep, sum_masks.new_zeros(1))

        scores, labels, _, keep_inds = mask_matrix_nms_onnx(
            masks,
            cls_labels,
            cls_scores,
            mask_area=sum_masks,
            nms_pre=cfg.nms_pre,
            max_num=cfg.max_per_img,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            filter_thr=cfg.filter_thr,
        )

        mask_preds = mask_preds[keep_inds].unsqueeze(0)
        post_params = get_post_processing_params(ctx.cfg)
        export_postprocess_mask = post_params.get("export_postprocess_mask", True)
        if export_postprocess_mask:
            upsampled_size = (featmap_size[0] * self.mask_stride, featmap_size[1] * self.mask_stride)
            mask_preds = F.interpolate(mask_preds, size=upsampled_size, mode="bilinear")
            bboxes = scores.new_zeros(batch_size, scores.shape[-1], 4)
        else:
            bboxes = scores.new_zeros(batch_size, scores.shape[-1], 2)
            # full screen box so we can postprocess mask outside the model
            bboxes = torch.cat(
                [bboxes, bboxes.new_full((*bboxes.shape[:2], 1), w), bboxes.new_full((*bboxes.shape[:2], 1), h)], dim=-1
            )

        labels = labels.reshape(batch_size, -1)
        dets = torch.cat([bboxes, scores.reshape(batch_size, -1, 1)], dim=-1)

        return dets, labels, mask_preds
