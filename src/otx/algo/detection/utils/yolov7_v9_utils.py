# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""Utils for YOLOv7 and v9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t
from torchvision.ops import batched_nms

if TYPE_CHECKING:
    from otx.algo.detection.detectors import SingleStageDetector


def auto_pad(kernel_size: _size_2_t, dilation: _size_2_t = 1, **kwargs) -> tuple[int, int]:
    """Auto Padding for the convolution blocks."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


def round_up(x: int | Tensor, div: int = 1) -> int | Tensor:
    """Rounds up `x` to the bigger-nearest multiple of `div`."""
    return x + (-x % div)


def calculate_iou(bbox1, bbox2, metrics="iou") -> Tensor:
    metrics = metrics.lower()
    EPS = 1e-9
    dtype = bbox1.dtype
    bbox1 = bbox1.to(torch.float32)
    bbox2 = bbox2.to(torch.float32)

    # Expand dimensions if necessary
    if bbox1.ndim == 2 and bbox2.ndim == 2:
        bbox1 = bbox1.unsqueeze(1)  # (Ax4) -> (Ax1x4)
        bbox2 = bbox2.unsqueeze(0)  # (Bx4) -> (1xBx4)
    elif bbox1.ndim == 3 and bbox2.ndim == 3:
        bbox1 = bbox1.unsqueeze(2)  # (BZxAx4) -> (BZxAx1x4)
        bbox2 = bbox2.unsqueeze(1)  # (BZxBx4) -> (BZx1xBx4)

    # Calculate intersection coordinates
    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(ymax_inter - ymin_inter, min=0)

    # Calculate area of each bbox
    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + EPS)
    if metrics == "iou":
        return iou.to(dtype)

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Calculate diagonal length of the smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(bbox1[..., 0], bbox2[..., 0])
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(bbox1[..., 1], bbox2[..., 1])
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if metrics == "diou":
        return diou.to(dtype)

    # Compute aspect ratio penalty term
    arctan = torch.atan((bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS),
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    alpha = v / (v - iou + 1 + EPS)
    # Compute CIoU
    ciou = diou - alpha * v
    return ciou.to(dtype)


def generate_anchors(image_size: list[int], strides: list[int]):
    """Find the anchor maps for each w, h.

    Args:
        image_size List: the image size of augmented image size
        strides List[8, 16, 32, ...]: the stride size for each predicted layer

    Returns:
        all_anchors [HW x 2]:
        all_scalers [HW]: The index of the best targets for each anchors
    """
    W, H = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = W // stride * H // stride
        scaler.append(torch.full((anchor_num,), stride))
        shift = stride // 2
        h = torch.arange(0, H, stride) + shift
        w = torch.arange(0, W, stride) + shift
        anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers


def transform_bbox(bbox: Tensor, indicator="xywh -> xyxy"):
    data_type = bbox.dtype
    in_type, out_type = indicator.replace(" ", "").split("->")

    if in_type not in ["xyxy", "xywh", "xycwh"] or out_type not in ["xyxy", "xywh", "xycwh"]:
        raise ValueError("Invalid input or output format")

    if in_type == "xywh":
        x_min = bbox[..., 0]
        y_min = bbox[..., 1]
        x_max = bbox[..., 0] + bbox[..., 2]
        y_max = bbox[..., 1] + bbox[..., 3]
    elif in_type == "xyxy":
        x_min = bbox[..., 0]
        y_min = bbox[..., 1]
        x_max = bbox[..., 2]
        y_max = bbox[..., 3]
    elif in_type == "xycwh":
        x_min = bbox[..., 0] - bbox[..., 2] / 2
        y_min = bbox[..., 1] - bbox[..., 3] / 2
        x_max = bbox[..., 0] + bbox[..., 2] / 2
        y_max = bbox[..., 1] + bbox[..., 3] / 2

    if out_type == "xywh":
        bbox = torch.stack([x_min, y_min, x_max - x_min, y_max - y_min], dim=-1)
    elif out_type == "xyxy":
        bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    elif out_type == "xycwh":
        bbox = torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min], dim=-1)

    return bbox.to(dtype=data_type)


class Vec2Box:
    def __init__(
        self,
        detector: SingleStageDetector,
        image_size: tuple[int, int],
        strides: list[int] | None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.strides = strides if strides else self.create_auto_anchor(detector, image_size)
        self.update(image_size, device)

    def create_auto_anchor(self, detector: SingleStageDetector, image_size: tuple[int, int]) -> list[int]:
        dummy_input = torch.zeros(1, 3, *image_size).to(self.device)
        dummy_main_preds, _ = detector(dummy_input)
        strides = []
        for predict_head in dummy_main_preds:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(image_size[1] // anchor_num[1])
        return strides

    def update(self, image_size: tuple[int, int], device: str | torch.device) -> None:
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.anchor_grid, self.scaler = anchor_grid.to(device), scaler.to(device)

    def __call__(self, predicts: list[Any]) -> tuple[Tensor, Tensor, Tensor]:
        preds_cls, preds_anc, preds_box = [], [], []
        for layer_output in predicts:
            pred_cls, pred_anc, pred_box = layer_output
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            preds_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
            preds_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_anc = torch.concat(preds_anc, dim=1)
        preds_box = torch.concat(preds_box, dim=1)

        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
        return preds_cls, preds_anc, preds_box


def bbox_nms(
    cls_dist: Tensor,
    bbox: Tensor,
    min_confidence: float = 0.05,
    min_iou: float = 0.9,
    confidence: Tensor | None = None,
):
    cls_dist = cls_dist.sigmoid() * (1 if confidence is None else confidence)

    # filter class by confidence
    cls_val, cls_idx = cls_dist.max(dim=-1, keepdim=True)
    valid_mask = cls_val > min_confidence
    valid_cls = cls_idx[valid_mask].float()
    valid_con = cls_val[valid_mask].float()
    valid_box = bbox[valid_mask.repeat(1, 1, 4)].view(-1, 4)

    batch_idx, *_ = torch.where(valid_mask)
    nms_idx = batched_nms(valid_box, valid_cls, batch_idx, min_iou)
    predicts_nms = []
    for idx in range(cls_dist.size(0)):
        instance_idx = nms_idx[idx == batch_idx[nms_idx]]

        predict_nms = torch.cat(
            [valid_cls[instance_idx][:, None], valid_box[instance_idx], valid_con[instance_idx][:, None]],
            dim=-1,
        )

        predicts_nms.append(predict_nms)
    return predicts_nms


def set_info_into_module(layer_dict: dict[str, Any]) -> nn.Module:
    layer = layer_dict.pop("module")
    for k, v in layer_dict.items():
        setattr(layer, k, v)
    return layer
