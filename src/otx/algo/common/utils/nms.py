# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmcv.ops.nms and mmdeploy.mmcv.ops.nms.

Reference :
    - https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/nms.py
    - https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/mmcv/ops/nms.py
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from otx.algo.common.utils.utils import dynamic_topk
from torch import Tensor
from torch.onnx import symbolic_helper as sym_help
from torchvision.ops.boxes import nms as torch_nms


class NMSop(torch.autograd.Function):
    """NMS operation."""

    @staticmethod
    def forward(
        ctx: Any,  # noqa: ARG004, ANN401
        bboxes: Tensor,
        scores: Tensor,
        iou_threshold: float,
        offset: int,  # noqa: ARG004
        score_threshold: float,
        max_num: int,
    ) -> Tensor:
        """Forward function."""
        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = scores > score_threshold
            bboxes, scores = bboxes[valid_mask], scores[valid_mask]
            valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)
        if bboxes.get_device() == -1 and bboxes.dtype == torch.bfloat16:  # torch nms kernel doesn't support bfloat16
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                inds = torch_nms(bboxes.float(), scores.float(), iou_threshold)
        else:
            inds = torch_nms(bboxes, scores, iou_threshold)

        if max_num > 0:
            inds = inds[:max_num]
        if is_filtering_by_score:
            inds = valid_inds[inds]
        return inds


def nms(
    boxes: Tensor | np.ndarray,
    scores: Tensor | np.ndarray,
    iou_threshold: float,
    offset: int = 0,
    score_threshold: float = 0,
    max_num: int = -1,
) -> tuple[Tensor | np.ndarray, Tensor | np.ndarray]:
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
        score_threshold (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.

    Returns:
        tuple: kept dets (boxes and scores) and indice, which always have
        the same data type as the input.

    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
        >>>                   [49.3, 32.9, 51.0, 35.3],
        >>>                   [49.2, 31.8, 51.0, 35.4],
        >>>                   [35.1, 11.5, 39.1, 15.7],
        >>>                   [35.6, 11.8, 39.3, 14.2],
        >>>                   [35.3, 11.5, 39.9, 14.5],
        >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)

    inds = NMSop.apply(boxes, scores, iou_threshold, offset, score_threshold, max_num)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    nms_cfg: dict | None = None,
    class_agnostic: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    # When using rotated boxes, only apply offsets on center.
    elif boxes.size(-1) == 5:
        # Strictly, the maximum coordinates of the rotating box
        # (x,y,w,h,a) should be calculated by polygon coordinates.
        # But the conversion from rotated box to polygon will
        # slow down the speed.
        # So we use max(x,y) + max(w,h) as max coordinate
        # which is larger than polygon max coordinate
        # max(x1, y1, x2, y2,x3, y3, x4, y4)
        max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
        boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]], dim=-1)
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    nms_op = nms_cfg_.pop("type", "nms")
    if isinstance(nms_op, str):
        nms_op = eval(nms_op)  # noqa: S307, PGH001

    split_thr = nms_cfg_.pop("split_thr", 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop("max_num", -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for idx in torch.unique(idxs):
            mask = (idxs == idx).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def multiclass_nms(
    boxes: Tensor,
    scores: Tensor,
    max_output_boxes_per_class: int = 1000,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = -1,
    output_index: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/mmcv/ops/nms.py#L267-L309

    This function helps exporting to onnx with batch and multiclass NMS op. It
    only supports class-agnostic detection results. That is, the scores is of
    shape (N, num_bboxes, num_classes) and the boxes is of shape (N, num_boxes,
    4).
    """
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]
    topk_inds = None
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = dynamic_topk(max_scores, pre_top_k)
        batch_inds = torch.arange(batch_size, device=scores.device).view(-1, 1).long()
        boxes = boxes[batch_inds, topk_inds, :]
        scores = scores[batch_inds, topk_inds, :]

    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSop.apply(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

    return _select_nms_index(
        scores,
        boxes,
        selected_indices,
        batch_size,
        keep_top_k=keep_top_k,
        pre_inds=topk_inds,
        output_index=output_index,
    )


def _select_nms_index(
    scores: torch.Tensor,
    boxes: torch.Tensor,
    nms_index: torch.Tensor,
    batch_size: int,
    keep_top_k: int = -1,
    pre_inds: torch.Tensor = None,
    output_index: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform NMS output.

    Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/mmcv/ops/nms.py#L186-L264

    Args:
        scores (Tensor): The detection scores of shape
            [N, num_classes, num_boxes].
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        nms_index (Tensor): NMS output of bounding boxes indexing.
        batch_size (int): Batch size of the input image.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        pre_inds (Tensor): The pre-topk indices of boxes before nms.
            Defaults to None.
        return_index (bool): Whether to return indices of original bboxes.
            Defaults to False.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    # index by nms output
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    boxes = boxes[batch_inds, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=1)

    # batch all
    batched_dets = dets.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_template = torch.arange(0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device)
    batched_dets = batched_dets.where(
        (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
        batched_dets.new_zeros(1),
    )

    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    batched_labels = batched_labels.where((batch_inds == batch_template.unsqueeze(1)), batched_labels.new_ones(1) * -1)

    new_batch_size = batched_dets.shape[0]

    # expand tensor to eliminate [0, ...] tensor
    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((new_batch_size, 1, 5))), 1)
    batched_labels = torch.cat((batched_labels, batched_labels.new_zeros((new_batch_size, 1))), 1)
    if output_index and pre_inds is not None:
        # batch all
        pre_inds = pre_inds[batch_inds, box_inds]
        pre_inds = pre_inds.unsqueeze(0).repeat(batch_size, 1)
        pre_inds = pre_inds.where((batch_inds == batch_template.unsqueeze(1)), pre_inds.new_zeros(1))
        pre_inds = torch.cat((pre_inds, -pre_inds.new_ones((new_batch_size, 1))), 1)
    # sort
    is_use_topk = keep_top_k > 0 and (torch.onnx.is_in_onnx_export() or keep_top_k < batched_dets.shape[1])
    if is_use_topk:
        _, topk_inds = dynamic_topk(batched_dets[:, :, -1], keep_top_k, dim=1)
    else:
        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
    topk_batch_inds = torch.arange(batch_size, dtype=topk_inds.dtype, device=topk_inds.device).view(-1, 1)
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]
    if output_index:
        if pre_inds is not None:
            topk_inds = pre_inds[topk_batch_inds, topk_inds, ...]
        return batched_dets, batched_labels, topk_inds
    # slice and recover the tensor
    return batched_dets, batched_labels


class ONNXNMSop(torch.autograd.Function):
    """Create onnx::NonMaxSuppression op.

    Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/mmcv/ops/nms.py#L14-L102

    NMS in mmcv only supports one class with no batch info. This class assists
    in exporting NMS of ONNX's definition.
    """

    @staticmethod
    def forward(
        ctx,  # noqa: ANN001, ARG004
        boxes: Tensor,
        scores: Tensor,
        max_output_boxes_per_class: int,
        iou_threshold: float,
        score_threshold: float,
    ) -> Tensor:
        """Get NMS output indices.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            Tensor: Selected indices of boxes. 2-D tensor of shape
            (num_selected_indices, 3) with each row of
            [batch_index, class_index, box_index].
        """
        batch_size, num_class, _ = scores.shape

        score_threshold = float(score_threshold)
        iou_threshold = float(iou_threshold)
        indices = []
        for batch_id in range(batch_size):
            for cls_id in range(num_class):
                _boxes = boxes[batch_id, ...]
                # score_threshold=0 requires scores to be contiguous
                _scores = scores[batch_id, cls_id, ...].contiguous()
                _, box_inds = nms(
                    _boxes,
                    _scores,
                    iou_threshold,
                    offset=0,
                    score_threshold=score_threshold,
                    max_num=max_output_boxes_per_class,
                )
                batch_inds = torch.zeros_like(box_inds) + batch_id
                cls_inds = torch.zeros_like(box_inds) + cls_id
                indices.append(torch.stack([batch_inds, cls_inds, box_inds], dim=-1))
        return torch.cat(indices)

    @staticmethod
    def symbolic(
        g: torch.onnx._internal.git_utils.GraphContext,  # noqa: SLF001
        boxes: Tensor,
        scores: Tensor,
        max_output_boxes_per_class: int,
        iou_threshold: float,
        score_threshold: float,
    ) -> Callable:
        """Symbolic function for onnx::NonMaxSuppression.

        Args:
            g (Graph): The traced onnx graph.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            NonMaxSuppression op for onnx.
        """
        if not sym_help._is_value(max_output_boxes_per_class):  # noqa: SLF001
            max_output_boxes_per_class = g.op(
                "Constant",
                value_t=torch.tensor(max_output_boxes_per_class, dtype=torch.long),
            )

        if not sym_help._is_value(iou_threshold):  # noqa: SLF001
            iou_threshold = g.op("Constant", value_t=torch.tensor([iou_threshold], dtype=torch.float))

        if not sym_help._is_value(score_threshold):  # noqa: SLF001
            score_threshold = g.op("Constant", value_t=torch.tensor([score_threshold], dtype=torch.float))
        return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
