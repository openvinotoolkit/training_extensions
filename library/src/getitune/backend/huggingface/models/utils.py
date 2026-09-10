# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Hugging Face model wrappers.

Reproject ground truth (boxes and masks) into the model's padded input canvas
so predictions and ground truth can be compared in the same coordinate space.
``_traceable_masks_to_boxes`` is the ONNX/OpenVINO-trace-safe replacement for
``torchvision.ops.masks_to_boxes``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as f

if TYPE_CHECKING:
    from getitune.data.entity.base import ImageInfo


def reproject_boxes_to_input_space(boxes: torch.Tensor, img_info: ImageInfo) -> torch.Tensor:
    """Move xyxy boxes from original image coordinates into the model's input canvas.

    Args:
        boxes: ``(N, 4)`` xyxy boxes in original image coordinates.
        img_info: Carries the ``scale_factor`` and ``padding`` the ``Resize``
            augmentation applied to this sample.

    Returns:
        ``(N, 4)`` boxes in the model's padded input coordinate space.

    Raises:
        ValueError: If ``img_info.scale_factor`` is unset (``None``), e.g.
            after a crop the ``Resize`` transform never ran.
    """
    if boxes.numel() == 0:
        return boxes
    if img_info.scale_factor is None:
        msg = "img_info.scale_factor is None; cannot reproject boxes without a Resize transform having run."
        raise ValueError(msg)

    scale_h, scale_w = img_info.scale_factor
    pad_left, pad_top, _, _ = img_info.padding
    reprojected = boxes.clone()
    reprojected[:, 0::2] = boxes[:, 0::2] * scale_w + pad_left
    reprojected[:, 1::2] = boxes[:, 1::2] * scale_h + pad_top
    return reprojected


def reproject_masks_to_input_space(
    masks: torch.Tensor,
    img_info: ImageInfo,
    input_size: tuple[int, int],
) -> torch.Tensor:
    """Move binary masks from original image coordinates into the model's input canvas.

    Resizes with nearest-neighbor interpolation to preserve binary values,
    then applies the same padding the ``Resize`` augmentation used.

    Args:
        masks: ``(N, ori_h, ori_w)`` binary masks in original image coordinates.
        img_info: Carries the ``scale_factor`` and ``padding`` applied to this sample.
        input_size: ``(H, W)`` of the model's input canvas.

    Returns:
        ``(N, H, W)`` binary masks in the model's padded input coordinate space.

    Raises:
        ValueError: If ``img_info.scale_factor`` is unset (``None``).
    """
    if masks.numel() == 0:
        return torch.zeros((0, *input_size), dtype=torch.bool, device=masks.device)
    if img_info.scale_factor is None:
        msg = "img_info.scale_factor is None; cannot reproject masks without a Resize transform having run."
        raise ValueError(msg)

    scale_h, scale_w = img_info.scale_factor
    pad_left, pad_top, pad_right, pad_bottom = img_info.padding
    ori_h, ori_w = img_info.ori_shape
    new_h, new_w = round(ori_h * scale_h), round(ori_w * scale_w)

    resized = f.interpolate(masks.unsqueeze(1).float(), size=(new_h, new_w), mode="nearest").squeeze(1)
    padded = f.pad(resized, [pad_left, pad_right, pad_top, pad_bottom])
    height, width = input_size
    return padded[:, :height, :width] > 0.5


def _traceable_masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """A ``torchvision.ops.masks_to_boxes`` reimplementation that traces to ONNX.

    ``masks_to_boxes`` itself assigns into an empty-mask boolean index
    (``bounding_boxes[empty_masks] = 0``), which OpenVINO's ONNX frontend
    cannot convert (``Select`` op with mismatched, non-broadcastable shapes —
    a boolean-index assignment traces very differently from a elementwise
    ``where``). Replacing it with an explicit ``torch.where`` over a
    broadcastable condition produces the identical result and converts
    cleanly.
    """
    _n, h, w = masks.shape
    masks_bool = masks.bool()
    non_zero_rows = torch.any(masks_bool, dim=2)
    non_zero_cols = torch.any(masks_bool, dim=1)
    empty = ~torch.any(non_zero_rows, dim=1)

    non_zero_rows_f = non_zero_rows.float()
    non_zero_cols_f = non_zero_cols.float()

    y1 = non_zero_rows_f.argmax(dim=1)
    x1 = non_zero_cols_f.argmax(dim=1)
    y2 = (h - 1) - non_zero_rows_f.flip(dims=[1]).argmax(dim=1)
    x2 = (w - 1) - non_zero_cols_f.flip(dims=[1]).argmax(dim=1)

    boxes = torch.stack([x1, y1, x2, y2], dim=1).float()
    return torch.where(empty.unsqueeze(-1), torch.zeros_like(boxes), boxes)
