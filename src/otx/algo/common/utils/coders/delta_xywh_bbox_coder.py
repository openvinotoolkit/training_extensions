# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.task_modules.coders.delta_xywh_bbox_coder.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py
"""

from __future__ import annotations

import numpy as np
import torch
from otx.algo.detection.utils.utils import clip_bboxes
from torch import Tensor


class DeltaXYWHBBoxCoder:
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        encode_size (int): Encode size.
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    """

    def __init__(
        self,
        encode_size: int = 4,
        target_means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
        target_stds: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
        clip_border: bool = True,
        add_ctr_clamp: bool = False,
        ctr_clamp: int = 32,
    ) -> None:
        self.encode_size = encode_size
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Get box regression transformation deltas that can be used to transform the bboxes into the gt_bboxes.

        Args:
            bboxes (Tensor): Source boxes,
                e.g., object proposals.
            gt_bboxes (Tensor): Target of the
                transformation, e.g., ground-truth boxes.

        Returns:
            Tensor: Box transformation deltas
        """
        return bbox2delta(bboxes, gt_bboxes, self.means, self.stds)

    def decode(
        self,
        bboxes: Tensor,
        pred_bboxes: Tensor,
        max_shape: tuple[int, ...] | Tensor | tuple[tuple[int, ...], ...] | None = None,
        wh_ratio_clip: float = 16 / 1000,
    ) -> Tensor:
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (Tensor): Basic boxes. Shape
                (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            Tensor: Decoded boxes.
        """
        return delta2bbox(
            bboxes,
            pred_bboxes,
            self.means,
            self.stds,
            max_shape,
            wh_ratio_clip,
            self.clip_border,
            self.add_ctr_clamp,
            self.ctr_clamp,
        )

    def decode_export(
        self,
        bboxes: Tensor,
        pred_bboxes: Tensor,
        max_shape: tuple[int, ...] | Tensor | tuple[tuple[int, ...], ...] | None = None,
        wh_ratio_clip: float = 16 / 1000,
    ) -> Tensor:
        """Apply transformation `pred_bboxes` to `boxes`.

        Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py#L12-L46

        Args:
            bboxes (Tensor): Basic boxes. Shape
                (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            Tensor: Decoded boxes.
        """
        return delta2bbox_export(
            bboxes,
            pred_bboxes,
            self.means,
            self.stds,
            max_shape,
            wh_ratio_clip,
            self.clip_border,
            self.add_ctr_clamp,
            self.ctr_clamp,
        )


def bbox2delta(
    proposals: Tensor,
    gt: Tensor,
    means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    stds: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
) -> Tensor:
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (tuple[float, ...]): Denormalizing means for delta coordinates
        stds (tuple[float, ...]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    return deltas.sub_(means).div_(stds)


def delta2bbox(
    rois: Tensor,
    deltas: Tensor,
    means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    stds: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    max_shape: tuple[int, ...] | Tensor | tuple[tuple[int, ...], ...] | None = None,
    wh_ratio_clip: float = 16 / 1000,
    clip_border: bool = True,
    add_ctr_clamp: bool = False,
    ctr_clamp: int = 32,
) -> Tensor:
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.
        add_ctr_clamp (bool): Whether to add center clamp. When set to True,
            the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor.
            Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 4) or (N, 4), where 4
           represent tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524
    """
    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    # Compute width/height of each roi
    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = (rois_[:, :2] + rois_[:, 2:]) * 0.5
    pwh = rois_[:, 2:] - rois_[:, :2]

    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    return bboxes.reshape(num_bboxes, -1)


def delta2bbox_export(
    rois: Tensor,
    deltas: Tensor,
    means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    stds: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    max_shape: tuple[int, ...] | Tensor | tuple[tuple[int, ...], ...] | None = None,
    wh_ratio_clip: float = 16 / 1000,
    clip_border: bool = True,
    add_ctr_clamp: bool = False,
    ctr_clamp: int = 32,
) -> Tensor:
    """Rewrite `delta2bbox` for default backend.

    Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py#L53-L138

    Since the need of clip op with dynamic min and max, this function uses
    clip_bboxes function to support dynamic shape.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.
        add_ctr_clamp (bool): Whether to add center clamp. When set to True,
            the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor.
            Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Return:
        bboxes (Tensor): Boxes with shape (N, num_classes * 4) or (N, 4),
            where 4 represent tl_x, tl_y, br_x, br_y.
    """
    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    delta_shape = deltas.shape
    reshaped_deltas = deltas.view(delta_shape[:-1] + (-1, 4))
    denorm_deltas = reshaped_deltas * stds + means

    dxy = denorm_deltas[..., :2]
    dwh = denorm_deltas[..., 2:]

    # fix openvino on torch1.13
    xy1 = rois[..., :2].unsqueeze(2)
    xy2 = rois[..., 2:].unsqueeze(2)

    pxy = (xy1 + xy2) * 0.5
    pwh = xy2 - xy1
    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    # Use exp(network energy) to enlarge/shrink each roi
    half_gwh = pwh * dwh.exp() * 0.5
    # Use network energy to shift the center of each roi
    gxy = pxy + dxy_wh

    # Convert center-xy/width/height to top-left, bottom-right
    xy1 = gxy - half_gwh
    xy2 = gxy + half_gwh

    x1 = xy1[..., 0]
    y1 = xy1[..., 1]
    x2 = xy2[..., 0]
    y2 = xy2[..., 1]

    if clip_border and max_shape is not None:
        x1, y1, x2, y2 = clip_bboxes(x1, y1, x2, y2, max_shape)

    return torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
