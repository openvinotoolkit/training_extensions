# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geometry and bbox conversion utilities for Ultralytics data bridge."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as f


def tensor_space_affine(
    tensor_h: int,
    tensor_w: int,
    canvas_size: tuple[int, int],
    scale_factor: tuple[float, float] | None = None,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[float, float, float, float]:
    """Return ``(scale_x, scale_y, offset_x, offset_y)`` mapping canvas -> tensor space.

    Single source of truth for the letterbox transform: ``rescale_bboxes_to_tensor_space``
    and :func:`rescale_points_to_tensor_space` must apply the *same* mapping, or a
    sample's boxes and its polygons end up in different coordinate frames.

    Args:
        tensor_h: Height of the model input tensor.
        tensor_w: Width of the model input tensor.
        canvas_size: ``(H, W)`` the annotations currently live in.
        scale_factor: ``(scale_h, scale_w)`` from ``ImageInfo``, if known.
        padding: ``(left, top, right, bottom)`` letterbox padding.

    Returns:
        ``(scale_x, scale_y, offset_x, offset_y)``.
    """
    if scale_factor is not None:
        scale_h, scale_w = scale_factor
        return float(scale_w), float(scale_h), float(padding[0]), float(padding[1])
    canvas_h, canvas_w = canvas_size
    return tensor_w / canvas_w, tensor_h / canvas_h, 0.0, 0.0


def xyxy_abs_to_xywh_norm(
    bboxes: torch.Tensor | np.ndarray,
    img_w: int,
    img_h: int,
    canvas_size: tuple[int, int] | None = None,
    scale_factor: tuple[float, float] | None = None,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Convert absolute XYXY bboxes to normalised xywh (Ultralytics format).

    Args:
        bboxes: ``(N, 4)`` tensor or array in ``[x1, y1, x2, y2]`` pixel coords.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        canvas_size: Optional ``(H, W)`` bbox coordinate canvas before tensor resize.
        scale_factor: Optional ``(scale_h, scale_w)`` for letterbox resize.
        padding: Optional ``(left, top, right, bottom)`` letterbox padding.

    Returns:
        ``(N, 4)`` float32 array in ``[cx, cy, w, h]`` normalised by image size.
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    bboxes = bboxes.astype(np.float32).copy()

    if bboxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if canvas_size is not None and canvas_size != (img_h, img_w):
        bboxes = rescale_bboxes_to_tensor_space(
            bboxes,
            tensor_h=img_h,
            tensor_w=img_w,
            canvas_size=canvas_size,
            scale_factor=scale_factor,
            padding=padding,
        )

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


def rescale_bboxes_to_tensor_space(
    bboxes: torch.Tensor | np.ndarray,
    tensor_h: int,
    tensor_w: int,
    canvas_size: tuple[int, int],
    scale_factor: tuple[float, float] | None = None,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Rescale XYXY bboxes from original/canvas space to tensor pixel space.

    When ``scale_factor`` and ``padding`` are provided (from ``ImageInfo``),
    the transformation accounts for letterbox padding.  Otherwise a simple
    proportional rescale is applied.

    Args:
        bboxes: ``(N, 4)`` tensor or array in ``[x1, y1, x2, y2]`` canvas coords.
        tensor_h: Height of the model input tensor.
        tensor_w: Width of the model input tensor.
        canvas_size: ``(H, W)`` of the coordinate space the bboxes live in.
        scale_factor: ``(scale_h, scale_w)`` applied during resize.
        padding: ``(left, top, right, bottom)`` letterbox padding.

    Returns:
        ``(N, 4)`` float32 array in tensor pixel coordinates.
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    bboxes = bboxes.astype(np.float32).copy()

    if bboxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    sx, sy, ox, oy = tensor_space_affine(tensor_h, tensor_w, canvas_size, scale_factor, padding)
    bboxes[:, 0::2] = bboxes[:, 0::2] * sx + ox
    bboxes[:, 1::2] = bboxes[:, 1::2] * sy + oy
    return bboxes


def rescale_points_to_tensor_space(
    coords: torch.Tensor,
    tensor_h: int,
    tensor_w: int,
    canvas_size: tuple[int, int],
    scale_factor: tuple[float, float] | None = None,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> torch.Tensor:
    """Map ``(..., 2)`` xy coordinates from canvas space into tensor pixel space.

    NaN entries (polygon padding and ring separators) pass through unchanged,
    since an affine map propagates NaN.

    Args:
        coords: ``(..., 2)`` float tensor of ``(x, y)`` coordinates.
        tensor_h: Height of the model input tensor.
        tensor_w: Width of the model input tensor.
        canvas_size: ``(H, W)`` the coordinates currently live in.
        scale_factor: ``(scale_h, scale_w)`` applied during resize.
        padding: ``(left, top, right, bottom)`` letterbox padding.

    Returns:
        ``(..., 2)`` float32 tensor in tensor pixel coordinates.
    """
    out = coords.detach().to(torch.float32).clone()
    if out.numel() == 0:
        return out
    sx, sy, ox, oy = tensor_space_affine(tensor_h, tensor_w, canvas_size, scale_factor, padding)
    out[..., 0] = out[..., 0] * sx + ox
    out[..., 1] = out[..., 1] * sy + oy
    return out


def build_ratio_pad(
    ori_shape: tuple[int, int],
    img_shape: tuple[int, int],
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[tuple[float, float], tuple[int, int]]:
    """Derive ``ratio_pad`` from getitune ``ImageInfo`` fields.

    Ultralytics validators need ``ratio_pad = ((rh, rw), (pad_x, pad_y))``
    where ``rh = content_h / ori_h``, ``rw = content_w / ori_w``,
    ``pad_x`` is horizontal (left) padding and ``pad_y`` is vertical (top) padding.

    The ratios must reflect the *content area* scale (before padding), not the
    final padded dimensions.  This matches Ultralytics' ``scale_boxes`` which
    uses ``gain = ratio_pad[0][0]`` to undo the resize, and subtracts padding
    separately::

        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    Args:
        ori_shape: ``(H, W)`` of the original image before any transforms.
        img_shape: ``(H, W)`` after resize and padding (may include padding).
        padding: ``(left, top, right, bottom)`` padding applied after resize.

    Returns:
        ``((ratio_h, ratio_w), (pad_x, pad_y))`` tuple.
    """
    ori_h, ori_w = ori_shape
    new_h, new_w = img_shape

    # Subtract padding to get the content area dimensions.
    pad_left, pad_top, pad_right, pad_bottom = padding
    content_h = new_h - pad_top - pad_bottom
    content_w = new_w - pad_left - pad_right

    # Avoid division-by-zero for degenerate images.
    rh = content_h / ori_h if ori_h > 0 else 1.0
    rw = content_w / ori_w if ori_w > 0 else 1.0

    return (rh, rw), (pad_left, pad_top)


def scale_masks_to_letterbox(masks: torch.Tensor, ori_h: int, ori_w: int, imgsz: int) -> torch.Tensor:
    """Transform binary masks from original image coords to letterbox-padded coords.

    Analogous to ``scale_boxes_to_letterbox`` but for spatial mask tensors.
    Resizes each mask with nearest-neighbor interpolation to preserve binary
    values, then center-pads to ``(imgsz, imgsz)``.

    Args:
        masks: ``(N, ori_h, ori_w)`` binary mask tensor.
        ori_h: Original image height.
        ori_w: Original image width.
        imgsz: Model input size (square).

    Returns:
        ``(N, imgsz, imgsz)`` binary mask tensor in letterbox coords.
    """
    if masks.numel() == 0:
        return torch.zeros((0, imgsz, imgsz), dtype=torch.bool, device=masks.device)

    scale = min(imgsz / ori_h, imgsz / ori_w)
    new_h = round(ori_h * scale)
    new_w = round(ori_w * scale)

    resized = f.interpolate(
        masks.unsqueeze(1).float(),
        size=(new_h, new_w),
        mode="nearest",
    ).squeeze(1)

    pad_y = (imgsz - new_h) // 2
    pad_x = (imgsz - new_w) // 2

    result = torch.zeros((masks.shape[0], imgsz, imgsz), dtype=torch.bool, device=masks.device)
    result[:, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized > 0.5
    return result


def letterbox_affine(ori_h: int, ori_w: int, imgsz: int) -> tuple[float, float, float]:
    """Return ``(scale, pad_x, pad_y)`` for an aspect-preserving centre-padded resize.

    Single source of truth for the test-time letterbox: boxes, polygons, and
    any future carrier must share it, or GT geometries land in different
    frames and the metric silently degrades instead of failing.

    Args:
        ori_h: Original image height.
        ori_w: Original image width.
        imgsz: Square model input size.

    Returns:
        ``(scale, pad_x, pad_y)`` — multiply then add.
    """
    scale = min(imgsz / ori_h, imgsz / ori_w)
    return scale, (imgsz - ori_w * scale) / 2.0, (imgsz - ori_h * scale) / 2.0


def scale_boxes_to_letterbox(boxes: torch.Tensor, ori_h: int, ori_w: int, imgsz: int) -> torch.Tensor:
    """Transform bounding boxes from original image coords to letterbox-padded coords.

    The DataModule's test augmentations use ``Resize(keep_aspect_ratio=True,
    center_padding=True, resize_targets=False)``, so target boxes remain in
    the original image coordinate space.  YOLO predictions, however, are in
    the letterbox-padded ``imgsz x imgsz`` space.  This method applies the
    same scale + center-pad offset to align the two.

    Args:
        boxes: ``(N, 4)`` tensor of xyxy boxes in original coords.
        ori_h: Original image height.
        ori_w: Original image width.
        imgsz: Model input size (square).

    Returns:
        Transformed ``(N, 4)`` tensor in letterbox coords.
    """
    if boxes.numel() == 0:
        return boxes
    scale, pad_x, pad_y = letterbox_affine(ori_h, ori_w, imgsz)
    scaled = boxes.clone()
    scaled[:, 0::2] = boxes[:, 0::2] * scale + pad_x
    scaled[:, 1::2] = boxes[:, 1::2] * scale + pad_y
    return scaled


def scale_points_to_letterbox(coords: torch.Tensor, ori_h: int, ori_w: int, imgsz: int) -> torch.Tensor:
    """Transform ``(..., 2)`` xy coordinates from original image coords to letterbox coords.

    The polygon analogue of :func:`scale_boxes_to_letterbox`.  NaN padding and
    ring separators pass through unchanged, so the result can be handed
    straight to ``rasterize``.

    Args:
        coords: ``(..., 2)`` float tensor of ``(x, y)`` in original coords.
        ori_h: Original image height.
        ori_w: Original image width.
        imgsz: Square model input size.

    Returns:
        ``(..., 2)`` float32 tensor in letterbox coords.
    """
    out = coords.detach().to(torch.float32).clone()
    if out.numel() == 0:
        return out
    scale, pad_x, pad_y = letterbox_affine(ori_h, ori_w, imgsz)
    out[..., 0] = out[..., 0] * scale + pad_x
    out[..., 1] = out[..., 1] * scale + pad_y
    return out


def points_abs_to_norm(coords: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Normalise ``(..., 2)`` pixel coordinates to ``[0, 1]`` (Ultralytics ``segments``)."""
    out = coords.clone()
    if out.numel() == 0:
        return out
    out[..., 0] /= img_w
    out[..., 1] /= img_h
    return out
