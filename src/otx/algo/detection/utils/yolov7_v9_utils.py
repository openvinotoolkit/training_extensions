# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""Utils for YOLOv7 and v9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange
from torch import Tensor, nn
from torchvision.ops import batched_nms

if TYPE_CHECKING:
    from otx.algo.detection.detectors import SingleStageDetector


def auto_pad(kernel_size: int | tuple[int, int], dilation: int | tuple[int, int] = 1, **kwargs) -> tuple[int, int]:  # noqa: ARG001
    """Auto Padding for the convolution blocks.

    Args:
        kernel_size (int | tuple[int, int]): The kernel size of the convolution block.
        dilation (int | tuple[int, int]): The dilation rate of the convolution block. Defaults to 1.

    Returns:
        tuple[int, int]: The padding size for the convolution block.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return pad_h, pad_w


def round_up(x: int | Tensor, div: int = 1) -> int | Tensor:
    """Rounds up `x` to the bigger-nearest multiple of `div`.

    Args:
        x (int | Tensor): The input value.
        div (int): The divisor value. Defaults to 1.

    Returns:
        int | Tensor: The rounded up value.
    """
    return x + (-x % div)


def generate_anchors(image_size: tuple[int, int], strides: list[int]) -> tuple[Tensor, Tensor]:
    """Find the anchor maps for each height and width.

    Args:
        image_size (tuple[int, int]): the image size of augmented image size.
        strides list[int]: the stride size for each predicted layer.

    Returns:
        tuple[Tensor, Tensor]: The anchor maps with (HW x 2) and the scaler maps with (HW,).
    """
    height, width = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = width // stride * height // stride
        scaler.append(torch.full((anchor_num,), stride))
        shift = stride // 2
        h = torch.arange(0, height, stride) + shift
        w = torch.arange(0, width, stride) + shift
        anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers


def transform_bbox(bbox: Tensor, indicator: str = "xywh -> xyxy") -> Tensor:
    """Transform the bounding box format.

    TODO (sungchul): replace it with transform utils.

    Args:
        bbox (Tensor): The bounding box tensor with (N, 4) shape.
        indicator (str): The indicator for input and output format. Defaults to "xywh -> xyxy".

    Returns:
        Tensor: The transformed bounding box tensor.
    """
    data_type = bbox.dtype
    in_type, out_type = indicator.replace(" ", "").split("->")

    if in_type not in ["xyxy", "xywh", "xycwh"] or out_type not in ["xyxy", "xywh", "xycwh"]:
        msg = "Invalid input or output format"
        raise ValueError(msg)

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
    """Convert the vector to bounding box.

    Args:
        detector (SingleStageDetector): The single stage detector instance.
        image_size (tuple[int, int]): The image size.
        strides (list[int] | None): The strides for each predicted layer. Defaults to None.
        device (str): The device to use. Defaults to "cpu".
    """

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
        """Create the auto anchor for the given detector.

        Args:
            detector (SingleStageDetector): The single stage detector instance.
            image_size (tuple[int, int]): The image size.

        Returns:
            list[int]: The strides for each predicted layer.
        """
        dummy_input = torch.zeros(1, 3, *image_size).to(self.device)
        dummy_main_preds, _ = detector(dummy_input)
        strides = []
        for predict_head in dummy_main_preds:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(image_size[1] // anchor_num[1])
        return strides

    def update(self, image_size: tuple[int, int], device: str | torch.device) -> None:
        """Update the anchor grid and scaler.

        Args:
            image_size (tuple[int, int]): The image size.
            device (str | torch.device): The device to use.
        """
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.anchor_grid, self.scaler = anchor_grid.to(device), scaler.to(device)

    def __call__(self, predicts: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Convert the vector to bounding box.

        Args:
            predicts (tuple[Tensor, Tensor, Tensor]): The list of prediction results.

        Returns:
            tuple[Tensor, Tensor, Tensor]: The converted results.
        """
        preds_cls, preds_anc, preds_box = [], [], []
        for layer_output in predicts:
            pred_cls, pred_anc, pred_box = layer_output
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            preds_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
            preds_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_anc = torch.concat(preds_anc, dim=1)
        preds_box = torch.concat(preds_box, dim=1)

        pred_lt_rb = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_lt_rb.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
        return preds_cls, preds_anc, preds_box


def bbox_nms(
    cls_dist: Tensor,
    bbox: Tensor,
    min_confidence: float = 0.05,
    min_iou: float = 0.9,
    confidence: Tensor | None = None,
) -> list[Tensor]:
    """Apply NMS to the bounding box.

    Args:
        cls_dist (Tensor): The class distribution tensor.
        bbox (Tensor): The bounding box tensor.
        min_confidence (float): The minimum confidence to filter. Defaults to 0.05.
        min_iou (float): The minimum IoU to filter. Defaults to 0.9.
        confidence (Tensor | None): The confidence tensor. Defaults to None.

    Returns:
        list[Tensor]: The list of predicted bounding boxes.
    """
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


def set_info_into_instance(layer_dict: dict[str, Any]) -> nn.Module:
    """Set the information into the instance.

    Args:
        layer_dict (dict[str, Any]): The dictionary of instance with given information.

    Returns:
        nn.Module: The instance with given information.
    """
    layer = layer_dict.pop("module")
    for k, v in layer_dict.items():
        setattr(layer, k, v)
    return layer
