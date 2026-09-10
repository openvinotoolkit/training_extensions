# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for build function."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as f


def get_default_num_async_infer_requests() -> int:
    """Returns a default number of infer request for OV models."""
    import os

    number_requests = os.cpu_count()
    number_requests = max(1, int(number_requests / 2)) if number_requests is not None else 1
    msg = f"""Set the default number of OpenVINO inference requests to {number_requests}.
            You can specify the value in config."""
    warnings.warn(msg, stacklevel=1)
    return number_requests


def rescale_bboxes_to_original(
    bboxes_data: torch.Tensor,
    img_shape: tuple[int, int],
    ori_shape: tuple[int, int],
    padding: tuple[int, int, int, int],
    scale_factor: tuple[float, float] | None,
) -> torch.Tensor:
    """Rescale predicted bounding boxes from model input coordinates to original image coordinates.

    Handles two preprocessing cases:
    1. Letterbox (aspect-ratio resize + padding): undo padding, then divide by scale_factor, then clamp.
    2. Simple resize (no padding): multiply by ori/img ratio.

    Args:
        bboxes_data: Tensor of shape (N, 4+) with bounding boxes in XYXY format.
        img_shape: (H, W) of the preprocessed model input image.
        ori_shape: (H, W) of the original image.
        padding: (left, top, right, bottom) padding applied during preprocessing.
        scale_factor: (scale_h, scale_w) applied during preprocessing, or None.

    Returns:
        The same tensor with coordinates mapped to ori_shape space.
    """
    img_h, img_w = img_shape
    ori_h, ori_w = ori_shape

    if (img_h, img_w) == (ori_h, ori_w) or bboxes_data.numel() == 0:
        return bboxes_data

    if padding != (0, 0, 0, 0):
        if scale_factor is None:
            msg = (
                "Non-zero padding with scale_factor=None is invalid. "
                "This indicates a preprocessing pipeline bug — padding implies resize, which must set scale_factor."
            )
            raise ValueError(msg)
        # Letterbox: undo padding then undo scale
        pad_left, pad_top = float(padding[0]), float(padding[1])
        scale_h, scale_w = float(scale_factor[0]), float(scale_factor[1])
        bboxes_data[:, 0::2] -= pad_left
        bboxes_data[:, 1::2] -= pad_top
        bboxes_data[:, 0::2] /= scale_w
        bboxes_data[:, 1::2] /= scale_h
        bboxes_data[:, 0::2].clamp_(0, ori_w)
        bboxes_data[:, 1::2].clamp_(0, ori_h)
        return bboxes_data

    # Simple resize (no padding)
    scale_x = ori_w / img_w
    scale_y = ori_h / img_h
    bboxes_data[:, 0::2] *= scale_x
    bboxes_data[:, 1::2] *= scale_y

    return bboxes_data


def rescale_masks_to_original(
    masks: torch.Tensor,
    img_shape: tuple[int, int],
    ori_shape: tuple[int, int],
    padding: tuple[int, int, int, int],
) -> torch.Tensor:
    """Rescale predicted binary masks from model input coordinates to original image coordinates.

    Handles two preprocessing cases:
    1. Letterbox (aspect-ratio resize + padding): crop padding to get content, then resize to ori_shape.
    2. Simple resize (no padding): resize directly to ori_shape.

    Args:
        masks: Tensor of shape (N, img_H, img_W) with binary masks (uint8 0/1).
        img_shape: (H, W) of the preprocessed model input image.
        ori_shape: (H, W) of the original image.
        padding: (left, top, right, bottom) padding applied during preprocessing.

    Returns:
        Tensor of shape (N, ori_H, ori_W) with masks mapped to ori_shape space.
    """
    img_h, img_w = img_shape
    ori_h, ori_w = ori_shape

    if masks.numel() == 0:
        return masks.new_zeros((masks.shape[0], ori_h, ori_w), dtype=masks.dtype)

    if (img_h, img_w) == (ori_h, ori_w):
        return masks

    if padding != (0, 0, 0, 0):
        # Letterbox: crop padding to get the content region, then resize to ori_shape.
        # Computing content dims from img_shape - padding is exact (no rounding issues).
        pad_left, pad_top, pad_right, pad_bottom = padding
        content_h = img_h - pad_top - pad_bottom
        content_w = img_w - pad_left - pad_right
        masks = masks[:, pad_top : pad_top + content_h, pad_left : pad_left + content_w]

    # Resize masks to ori_shape using bilinear interpolation
    # f.interpolate expects (N, C, H, W) input
    masks_4d = masks.unsqueeze(1).float()  # (N, 1, H, W)
    return (f.interpolate(masks_4d, size=(ori_h, ori_w), mode="bilinear", align_corners=False).squeeze(1) > 0.5).to(
        torch.uint8
    )
