# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom RTMDetInsSepBNHead for OTX RTMDet-InstSeg instance segmentation models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from mmcv.ops import RoIAlign, batched_nms
from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.codebase.mmdet.models.dense_heads.rtmdet_ins_head import _parse_dynamic_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops.nms import multiclass_nms
from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, get_box_wh, scale_boxes
from mmengine.config import ConfigDict

if TYPE_CHECKING:
    from mmengine.structures import InstanceData


@MODELS.register_module()
class CustomRTMDetInsSepBNHead(RTMDetInsSepBNHead):
    """Custom RTMDet instance segmentation head.

    Note: In comparison to the original RTMDetInsSepBNHead, this class overrides the _bbox_mask_post_process
    to conduct mask post-processing by chunking the masks into smaller chunks and processing them individually.
    This approach mitigates the risk of running out of memory, particularly when handling a large number of masks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_align = RoIAlign(output_size=(28, 28))

    def mask_postprocess(
        self,
        mask_logits: torch.Tensor,
        img_h: int,
        img_w: int,
        gpu_mem_limit: float = 1.0,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Postprocess mask logits to binary masks.

        Args:
            masks (_type_): Mask Logits with shape (B, N, H, W).
            img_h (int): Image height resizes to.
            img_w (int): Image width resizes to.
            gpu_mem_limit (float, optional): GPU memory limit in GB. Defaults to 1.0.
            threshold (float, optional): Threshold for binary masks. Defaults to 0.5.

        Returns:
            torch.Tensor: Binary masks with shape (B, N, img_h, img_w).
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
                F.interpolate(
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
        mask_feat: torch.Tensor,
        cfg: ConfigDict | dict,
        rescale: bool = False,
        with_nms: bool = True,
        img_meta: dict | None = None,
    ) -> InstanceData:
        """Bbox and mask post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            mask_feat (Tensor): Mask prototype features of a single image
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
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
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        if img_meta is None:
            img_meta = {}
            img_meta["scale_factor"] = [1.0, 1.0]

        if cfg is None:
            cfg = ConfigDict(
                nms_pre=300,
                mask_thr_binary=0.5,
                max_per_img=100,
                score_thr=0.05,
                nms=ConfigDict(type="nms", iou_threshold=0.6),
                min_bbox_size=0,
            )

        stride = self.prior_generator.strides[0][0]
        if rescale:
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, "score_factors"):
            score_factors = results.pop("score_factors")
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.min_bbox_size >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if not with_nms:
            msg = "with_nms must be True for RTMDet-Ins"
            raise RuntimeError(msg)

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            # NOTE: mmcv.batched_nms Ops does not support half precision bboxes
            if bboxes.dtype != torch.float32:
                bboxes = bboxes.float()
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[: cfg.max_per_img]

            # process masks
            mask_logits = self._mask_predict_by_feat_single(mask_feat, results.kernels, results.priors)
            mask_logits = F.interpolate(mask_logits.unsqueeze(0), scale_factor=stride, mode="bilinear")

            if rescale:
                ori_h, ori_w = img_meta["ori_shape"][:2]
                masks = self.mask_postprocess(
                    mask_logits,
                    math.ceil(mask_logits.shape[-1] * scale_factor[1]),
                    math.ceil(mask_logits.shape[-2] * scale_factor[0]),
                    threshold=cfg.mask_thr_binary,
                )[..., :ori_h, :ori_w]
                masks = masks.squeeze(0)
            else:
                masks = mask_logits.sigmoid().squeeze(0)
                masks = masks > cfg.mask_thr_binary
            results.masks = masks
        else:
            h, w = img_meta["ori_shape"][:2] if rescale else img_meta["img_shape"][:2]
            results.masks = torch.zeros(
                size=(results.bboxes.shape[0], h, w),
                dtype=torch.bool,
                device=results.bboxes.device,
            )

        return results


def _custom_mask_predict_by_feat_single(
    self: CustomRTMDetInsSepBNHead,
    mask_feat: torch.Tensor,
    kernels: torch.Tensor,
    priors: torch.Tensor,
) -> torch.Tensor:
    """Decode mask with dynamic conv.

    Note: Prior Generator has cuda device set as default.
    However, this would cause some problems on CPU only devices.
    """
    num_inst = priors.shape[1]
    batch_size = priors.shape[0]
    hw = mask_feat.size()[-2:]
    # NOTE: had to force to set device in prior generator
    coord = self.prior_generator.single_level_grid_priors(hw, level_idx=0, device=mask_feat.device).to(mask_feat.device)
    coord = coord.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    priors = priors.unsqueeze(2)
    points = priors[..., :2]
    relative_coord = (points - coord).permute(0, 1, 3, 2) / (priors[..., 2:3] * 8)
    relative_coord = relative_coord.reshape(batch_size, num_inst, 2, hw[0], hw[1])

    mask_feat = torch.cat([relative_coord, mask_feat.unsqueeze(1).repeat(1, num_inst, 1, 1, 1)], dim=2)
    weights, biases = _parse_dynamic_params(self, kernels)

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


def _custom_nms_with_mask_static(
    self: CustomRTMDetInsSepBNHead,
    priors: torch.Tensor,
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    kernels: torch.Tensor,
    mask_feats: torch.Tensor,
    max_output_boxes_per_class: int = 1000,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = -1,
    mask_thr_binary: float = 0.5,  # noqa: ARG001
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrapper for `multiclass_nms` with ONNXRuntime.

    Note:
        Compared with the original _nms_with_mask_static, this function
        crops masks using RoIAlign and returns the cropped masks.

    Args:
        self: The instance of `RTMDetInsHead`.
        priors (Tensor): The prior boxes of shape [num_boxes, 4].
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
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
    mask_logits = _custom_mask_predict_by_feat_single(self, mask_feats, kernels, priors)
    stride = self.prior_generator.strides[0][0]
    mask_logits = F.interpolate(mask_logits, scale_factor=stride, mode="bilinear")
    masks = mask_logits.sigmoid()

    batch_index = (
        torch.arange(dets.size(0), device=dets.device).float().view(-1, 1, 1).expand(dets.size(0), dets.size(1), 1)
    )
    rois = torch.cat([batch_index, dets[..., :4]], dim=-1)
    cropped_masks = self.roi_align(masks, rois[0])
    cropped_masks = cropped_masks[torch.arange(cropped_masks.size(0)), torch.arange(cropped_masks.size(0))]
    cropped_masks = cropped_masks.unsqueeze(0)
    return dets, labels, cropped_masks


@FUNCTION_REWRITER.register_rewriter(
    func_name="otx.algo.instance_segmentation.heads.custom_rtmdet_ins_head.CustomRTMDetInsSepBNHead.predict_by_feat",
)
def rtmdet_ins_head__predict_by_feat(
    self: CustomRTMDetInsSepBNHead,
    cls_scores: list[torch.Tensor],
    bbox_preds: list[torch.Tensor],
    kernel_preds: list[torch.Tensor],
    mask_feat: torch.Tensor,
    score_factors: list[torch.Tensor] | None = None,  # noqa: ARG001
    batch_img_metas: list[dict] | None = None,  # noqa: ARG001
    cfg: ConfigDict | None = None,
    rescale: bool = False,  # noqa: ARG001
    with_nms: bool = True,  # noqa: ARG001
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rewrite `predict_by_feat` of `RTMDet-Ins` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

    Args:
        ctx: Context that contains original meta information.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
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
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
            represents the class label of the corresponding box.
    """
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

    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get("iou_threshold", post_params.iou_threshold)
    score_threshold = cfg.get("score_thr", post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get("max_per_img", post_params.keep_top_k)
    mask_thr_binary = cfg.get("mask_thr_binary", 0.5)

    return _custom_nms_with_mask_static(
        self,
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
        mask_thr_binary,
    )
