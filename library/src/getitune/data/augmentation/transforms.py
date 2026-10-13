# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom image transforms for getitune augmentation pipeline."""

from __future__ import annotations

import typing
from typing import Any, cast

import kornia.augmentation as K  # noqa: N812
import torch
import torchvision.transforms.v2 as tvt_v2
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F  # noqa: N812

from getitune.data.augmentation.cache import CacheableMixin, _CachedSample, _clone_for_cache
from getitune.data.augmentation.kernels import (
    _resize_image_info,
    _resized_crop_image_info,
)
from getitune.data.entity.sample import BaseSample


class Resize(tvt_v2.Transform):
    """Resize transform based on torchvision.transforms.v2.

    Extends torchvision's Resize with optional control over target resizing
    and aspect ratio preservation with padding.

    Args:
        size (int or tuple): Target size (height, width). If int, a square (size, size) is used.
        resize_targets (bool): If True, resize all targets (bboxes, masks, keypoints)
            along with the image. If False, resize only the image and leave targets unchanged.
            Defaults to True.
        keep_aspect_ratio (bool): If True, preserve the aspect ratio of the original image
            by resizing to fit within the target size and padding to reach exact target dimensions.
            Padding is applied to bottom-right only (matching YOLOX/DFine post-processing
            which assumes no left/top offset). Defaults to False.
        center_padding (bool): If True and ``keep_aspect_ratio`` is True, distribute
            padding equally on both sides (centered letterbox, matching Ultralytics
            LetterBox). If False, padding is applied to bottom-right only.
            Defaults to False.
        pad_value (int or tuple): Padding value for image. Defaults to 0.
        interpolation: Interpolation mode for images. Defaults to InterpolationMode.BILINEAR.
        antialias: Whether to apply antialiasing. Defaults to True.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        resize_targets: bool = True,
        keep_aspect_ratio: bool = False,
        center_padding: bool = False,
        pad_value: int | tuple[int, int, int] = 0,
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.resize_targets = resize_targets
        self.keep_aspect_ratio = keep_aspect_ratio
        self.center_padding = center_padding
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.antialias = antialias

    def _compute_resize_params(
        self,
        orig_h: int,
        orig_w: int,
    ) -> tuple[int, int, int, int, int, int]:
        """Compute resize dimensions and padding.

        Returns:
            new_h, new_w: Size after resize (before padding).
            pad_left, pad_top, pad_right, pad_bottom: Padding amounts.
        """
        target_h, target_w = self.size

        if not self.keep_aspect_ratio:
            return target_h, target_w, 0, 0, 0, 0

        # Compute scale to fit within target while preserving aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)

        # Compute padding to reach target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h

        if self.center_padding:
            # Centered letterbox (matching Ultralytics LetterBox): equal on both sides.
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
        else:
            # Bottom-right padding only (matching DFine/YOLOX post-processing
            # which assumes no left/top offset).
            pad_left = 0
            pad_right = pad_w
            pad_top = 0
            pad_bottom = pad_h

        return new_h, new_w, pad_left, pad_top, pad_right, pad_bottom

    def forward(self, *inputs: BaseSample) -> BaseSample:  # type: ignore[override]
        """Resize image and optionally targets, with optional aspect ratio preservation."""
        if len(inputs) > 1:
            msg = "Resize expects a single BaseSample input"
            raise ValueError(msg)
        sample: BaseSample = inputs[0]

        if not hasattr(sample, "image"):
            # Fallback: just resize the tensor directly
            return cast(
                "BaseSample",
                F.resize(
                    cast("torch.Tensor", sample),
                    size=list(self.size),
                    interpolation=self.interpolation,
                    antialias=self.antialias,
                ),
            )

        # Get original dimensions
        orig_h, orig_w = sample.image.shape[-2:]

        # Early exit: image already at target size — nothing to do
        if (orig_h, orig_w) == self.size:
            return sample

        # Compute resize and padding parameters
        new_h, new_w, pad_left, pad_top, pad_right, pad_bottom = self._compute_resize_params(orig_h, orig_w)

        # Resize image
        sample.image = F.resize(
            sample.image,
            size=[new_h, new_w],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        # Bilinear/bicubic interpolation can produce values slightly outside [0,1]
        if sample.image.is_floating_point():
            sample.image = sample.image.clamp(0, 1)
        # Apply padding if needed
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            # Normalise pad value to match the image's value range.
            is_float_image = sample.image.is_floating_point()
            if isinstance(self.pad_value, tuple):
                fill_value: float | int | list[float] = [
                    (v / 255.0 if is_float_image and v > 1.0 + 1e-5 else v) for v in self.pad_value
                ]
            else:
                fill_value = (
                    self.pad_value / 255.0 if is_float_image and self.pad_value > 1.0 + 1e-5 else self.pad_value
                )
            sample.image = F.pad(
                sample.image,
                padding=[pad_left, pad_top, pad_right, pad_bottom],
                fill=fill_value,
            )

        # Calculate scale factors for target transforms
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        # Resize/transform targets if requested
        if self.resize_targets:
            # Transform bounding boxes
            bboxes = getattr(sample, "bboxes", None)
            if bboxes is not None and len(bboxes) > 0:
                # Scale bboxes
                if isinstance(bboxes, tv_tensors.BoundingBoxes):
                    bboxes_data = bboxes.clone()
                else:
                    bboxes_data = bboxes.clone() if isinstance(bboxes, torch.Tensor) else torch.as_tensor(bboxes)

                # Apply scaling
                bboxes_data[..., 0::2] = bboxes_data[..., 0::2] * scale_x  # x coordinates
                bboxes_data[..., 1::2] = bboxes_data[..., 1::2] * scale_y  # y coordinates

                # Apply padding offset
                if pad_left > 0 or pad_top > 0:
                    bboxes_data[..., 0::2] = bboxes_data[..., 0::2] + pad_left  # x coordinates
                    bboxes_data[..., 1::2] = bboxes_data[..., 1::2] + pad_top  # y coordinates

                sample.bboxes = tv_tensors.BoundingBoxes(  # type: ignore[missing-attribute]
                    bboxes_data,
                    format=bboxes.format if isinstance(bboxes, tv_tensors.BoundingBoxes) else "XYXY",
                    canvas_size=self.size,
                )

            # Transform masks
            masks = getattr(sample, "masks", None)
            if masks is not None:
                if len(masks) > 0:
                    resized_masks = F.resize(
                        masks,
                        size=[new_h, new_w],
                        interpolation=F.InterpolationMode.NEAREST,
                        antialias=False,
                    )
                    # Pad masks
                    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                        resized_masks = F.pad(
                            resized_masks,
                            padding=[pad_left, pad_top, pad_right, pad_bottom],
                            fill=0,
                        )
                else:
                    # Empty mask: just reshape spatial dims to target size
                    resized_masks = masks.new_zeros((0, new_h + pad_top + pad_bottom, new_w + pad_left + pad_right))
                if resized_masks.shape[-2:] != sample.image.shape[-2:]:
                    msg = (
                        "Resized masks spatial dimensions must match the transformed image shape: "
                        f"{resized_masks.shape[-2:]} != {sample.image.shape[-2:]}"
                    )
                    raise RuntimeError(msg)
                sample.masks = (  # type: ignore[missing-attribute]
                    tv_tensors.Mask(resized_masks) if isinstance(masks, tv_tensors.Mask) else resized_masks
                )

            # Transform keypoints/points
            keypoints = getattr(sample, "keypoints", None)
            if keypoints is not None and isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.clone()
                # Scale keypoints (assuming format [..., x, y] or [..., x, y, visibility])
                keypoints[..., 0] = keypoints[..., 0] * scale_x + pad_left
                keypoints[..., 1] = keypoints[..., 1] * scale_y + pad_top
                sample.keypoints = keypoints  # type: ignore[missing-attribute]

        # Update img_info if available
        if hasattr(sample, "img_info") and sample.img_info is not None:
            # First update scale info based on resized (pre-pad) shape
            sample.img_info = _resize_image_info(sample.img_info, (new_h, new_w))
            # Then apply padding metadata if any
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                sample.img_info.padding = (pad_left, pad_top, pad_right, pad_bottom)
                sample.img_info.img_shape = (
                    new_h + pad_top + pad_bottom,
                    new_w + pad_left + pad_right,
                )
            if self.keep_aspect_ratio:
                # Store padding info for potential inverse transforms
                sample.img_info.pad_offset = (pad_left, pad_top, pad_right, pad_bottom)  # type: ignore[missing-attribute]
                # ImageInfo.scale_factor uses (height, width)
                sample.img_info.scale_factor = (scale_y, scale_x)
                sample.img_info.keep_ratio = True

        return sample


class CachedMosaic(CacheableMixin, tvt_v2.Transform):
    """Cached Mosaic augmentation with built-in random perspective crop.

    Combines four images into a 2x mosaic canvas, then applies a center-based
    random affine crop to produce an ``img_scale``-sized output.

    When mosaic is **skipped** (probability gate or cold cache) the sample is
    letterbox-resized to ``img_scale`` and the same random affine is applied
    (consistent non-mosaic path).

    The output size is **always** ``img_scale`` regardless of whether mosaic
    fires.

    Args:
        img_scale: Target output size ``(H, W)``.  Also the per-tile target for
            mosaic assembly.
        center_ratio_range: Range for the random mosaic centre.
        bbox_clip_border: Clip bboxes to the output boundary.
        pad_val: Fill value for canvas and border (0-255 scale, auto-normalised).
        p: Probability of applying mosaic.
        max_cached_images: Maximum cache size (>= 4).
        random_pop: Random eviction vs FIFO.
        scale: Scale factor for random perspective crop.  The random scale ``s``
            is sampled from ``[1 - scale, 1 + scale]``.  Defaults to 0.5.
        translate: Translate fraction for random perspective crop.  The crop
            centre shifts by up to ``± translate * img_scale`` pixels.
            Defaults to 0.1.
        area_thr: Minimum visible-area ratio for keeping a bbox after the
            perspective crop.  Lower values keep more small / partially
            occluded instances.  Defaults to 0.1.
    """

    def __init__(
        self,
        img_scale: tuple[int, int] | list[int] = (640, 640),
        center_ratio_range: tuple[float, float] = (0.5, 1.5),
        bbox_clip_border: bool = True,
        pad_val: float | tuple[float, float, float] = 114.0,
        p: float = 1.0,
        max_cached_images: int = 40,
        random_pop: bool = True,
        scale: float = 0.5,
        translate: float = 0.1,
        area_thr: float = 0.1,
    ) -> None:
        super().__init__()

        if not isinstance(img_scale, (tuple, list)):
            msg = "img_scale must be a tuple or list"
            raise TypeError(msg)
        if not 0 <= p <= 1.0:
            msg = f"probability must be in [0, 1], got {p}"
            raise ValueError(msg)
        if max_cached_images < 4:
            msg = f"max_cached_images must be >= 4, got {max_cached_images}"
            raise ValueError(msg)

        self.img_scale = tuple(img_scale)  # (H, W)
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = p
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.scale = scale
        self.translate = translate
        self.area_thr = area_thr
        self._init_cache()

    def _resize_keep_ratio(self, img: torch.Tensor, target_h: int, target_w: int) -> tuple[torch.Tensor, float]:
        """Resize CHW image keeping aspect ratio (FIT mode). Returns (resized, scale)."""
        _, h, w = img.shape
        scale = min(target_h / h, target_w / w)
        new_h = round(h * scale)
        new_w = round(w * scale)
        resized = F.resize(img, size=[new_h, new_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        return resized, scale

    def _compute_mosaic_params(
        self,
        loc: str,
        center_x: int,
        center_y: int,
        img_h: int,
        img_w: int,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int, int]:
        """Compute paste and crop coordinates for mosaic placement."""
        mosaic_h = self.img_scale[0] * 2
        mosaic_w = self.img_scale[1] * 2

        if loc == "top_left":
            x1_p = max(center_x - img_w, 0)
            y1_p = max(center_y - img_h, 0)
            x2_p = center_x
            y2_p = center_y
            x1_c = img_w - (x2_p - x1_p)
            y1_c = img_h - (y2_p - y1_p)
            x2_c = img_w
            y2_c = img_h
        elif loc == "top_right":
            x1_p = center_x
            y1_p = max(center_y - img_h, 0)
            x2_p = min(center_x + img_w, mosaic_w)
            y2_p = center_y
            x1_c = 0
            y1_c = img_h - (y2_p - y1_p)
            x2_c = min(img_w, x2_p - x1_p)
            y2_c = img_h
        elif loc == "bottom_left":
            x1_p = max(center_x - img_w, 0)
            y1_p = center_y
            x2_p = center_x
            y2_p = min(center_y + img_h, mosaic_h)
            x1_c = img_w - (x2_p - x1_p)
            y1_c = 0
            x2_c = img_w
            y2_c = min(img_h, y2_p - y1_p)
        else:  # bottom_right
            x1_p = center_x
            y1_p = center_y
            x2_p = min(center_x + img_w, mosaic_w)
            y2_p = min(center_y + img_h, mosaic_h)
            x1_c = 0
            y1_c = 0
            x2_c = min(img_w, x2_p - x1_p)
            y2_c = min(img_h, y2_p - y1_p)

        paste_coord = (x1_p, y1_p, x2_p, y2_p)
        crop_coord = (x1_c, y1_c, x2_c, y2_c)
        return paste_coord, crop_coord, x1_p - x1_c, y1_p - y1_c

    def _fill_val_normalised(self) -> float:
        fill = self.pad_val if isinstance(self.pad_val, (int, float)) else self.pad_val[0]
        return fill / 255.0 if fill > 1.0 + 1e-5 else fill

    def get_indexes(self, cache: list) -> list:
        """Get 3 random indexes from cache."""
        return [int(torch.randint(0, len(cache), (1,)).item()) for _ in range(3)]

    def _build_mosaic(
        self,
        inputs: _CachedSample,
        mix_results: list[_CachedSample],
        with_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Assemble the 4-image mosaic on a 2x canvas.

        Returns:
            mosaic_img: (C, 2*H, 2*W) canvas.
            mosaic_bboxes: (N, 4) XYXY in 2x canvas coords.
            mosaic_labels: (N,) class labels.
            mosaic_masks: (N, 2*H, 2*W) or None.
        """
        target_h, target_w = self.img_scale
        mosaic_h, mosaic_w = target_h * 2, target_w * 2

        center_x = int(torch.empty(1).uniform_(*self.center_ratio_range).item() * target_w)
        center_y = int(torch.empty(1).uniform_(*self.center_ratio_range).item() * target_h)

        device = inputs.image.device
        num_channels = inputs.image.shape[0]
        fill_val = self._fill_val_normalised()
        mosaic_img = torch.full((num_channels, mosaic_h, mosaic_w), fill_val, dtype=inputs.image.dtype, device=device)

        all_bboxes: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        samples = [inputs, *mix_results]

        for i, loc in enumerate(loc_strs):
            sample = samples[i]
            img_i = sample.image
            _, orig_h, orig_w = img_i.shape

            img_i, scale = self._resize_keep_ratio(img_i, target_h, target_w)
            _, new_h, new_w = img_i.shape

            paste_coord, crop_coord, pad_w, pad_h = self._compute_mosaic_params(
                loc,
                center_x,
                center_y,
                new_h,
                new_w,
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            mosaic_img[:, y1_p:y2_p, x1_p:x2_p] = img_i[:, y1_c:y2_c, x1_c:x2_c]

            # Scale and translate bboxes
            bboxes_i = sample.bboxes.float() * scale
            if bboxes_i.numel() > 0:
                bboxes_i = bboxes_i + bboxes_i.new_tensor([pad_w, pad_h, pad_w, pad_h])
            all_bboxes.append(bboxes_i)
            all_labels.append(sample.label)

            if with_mask:
                masks_i = getattr(sample, "masks", None)
                if masks_i is not None and len(masks_i) > 0:
                    s = min(target_h / orig_h, target_w / orig_w)
                    new_mh, new_mw = round(orig_h * s), round(orig_w * s)
                    masks_i = F.resize(
                        masks_i,
                        [new_mh, new_mw],
                        interpolation=F.InterpolationMode.NEAREST,
                        antialias=False,
                    )
                    n_masks = masks_i.shape[0]
                    mask_canvas = torch.zeros((n_masks, mosaic_h, mosaic_w), dtype=torch.uint8, device=device)
                    mask_canvas[:, y1_p:y2_p, x1_p:x2_p] = masks_i[:, y1_c:y2_c, x1_c:x2_c]
                    all_masks.append(mask_canvas)

        mosaic_bboxes = torch.cat(all_bboxes, dim=0)
        mosaic_labels = torch.cat(all_labels, dim=0)

        if self.bbox_clip_border and mosaic_bboxes.numel() > 0:
            mosaic_bboxes[..., 0::2] = mosaic_bboxes[..., 0::2].clamp(0, mosaic_w)
            mosaic_bboxes[..., 1::2] = mosaic_bboxes[..., 1::2].clamp(0, mosaic_h)

        # Filter degenerate bboxes
        if mosaic_bboxes.numel() > 0:
            w = mosaic_bboxes[:, 2] - mosaic_bboxes[:, 0]
            h = mosaic_bboxes[:, 3] - mosaic_bboxes[:, 1]
            valid = (w > 0) & (h > 0)
            mosaic_bboxes = mosaic_bboxes[valid]
            mosaic_labels = mosaic_labels[valid]
            if with_mask and all_masks:
                all_masks_cat = torch.cat(all_masks, dim=0)
                all_masks_cat = all_masks_cat[valid]
            else:
                all_masks_cat = None
        elif with_mask and all_masks:
            all_masks_cat = torch.cat(all_masks, dim=0)
        else:
            all_masks_cat = None

        mosaic_masks = all_masks_cat if (with_mask and all_masks) else None
        return mosaic_img.clamp(0, 1), mosaic_bboxes, mosaic_labels, mosaic_masks

    def _letterbox_resize(
        self,
        image: torch.Tensor,
        bboxes: torch.Tensor,
        masks: torch.Tensor | None,
        target_h: int,
        target_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Resize image with keep-ratio and center padding.

        Returns:
            image: (C, target_h, target_w) letterboxed.
            bboxes: (N, 4) XYXY in letterboxed coords.
            masks: (N, target_h, target_w) or None.
        """
        _, orig_h, orig_w = image.shape
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = round(orig_h * scale)
        new_w = round(orig_w * scale)

        fill_val = self._fill_val_normalised()

        resized = F.resize(image, size=[new_h, new_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        canvas = torch.full((image.shape[0], target_h, target_w), fill_val, dtype=image.dtype, device=image.device)
        canvas[:, pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        if bboxes.numel() > 0:
            bboxes = bboxes.float() * scale
            bboxes = bboxes + bboxes.new_tensor([pad_left, pad_top, pad_left, pad_top])

        if masks is not None and masks.numel() > 0:
            masks = F.resize(masks, [new_h, new_w], interpolation=F.InterpolationMode.NEAREST, antialias=False)
            mask_canvas = torch.zeros((masks.shape[0], target_h, target_w), dtype=masks.dtype, device=masks.device)
            mask_canvas[:, pad_top : pad_top + new_h, pad_left : pad_left + new_w] = masks
            masks = mask_canvas

        return canvas, bboxes, masks

    def _random_perspective_crop(
        self,
        image: torch.Tensor,
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor | None,
        out_h: int,
        out_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply center-based random affine crop.

        Reference: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py

        For mosaic input (2x canvas): crops a variable-size window from the center
        and resizes to (out_h, out_w).  For pass-through input (1x): applies
        random zoom/pan within the same canvas size.

        Args:
            image: (C, in_h, in_w) input image.
            bboxes: (N, 4) XYXY in input pixel coords.
            labels: (N,) class labels.
            masks: (N, in_h, in_w) or None.
            out_h: Output height.
            out_w: Output width.

        Returns:
            Tuple of (output_image, filtered_bboxes, filtered_labels, filtered_masks).
        """
        _, in_h, in_w = image.shape
        fill_val = self._fill_val_normalised()

        # Sample random scale and translation
        s = torch.empty(1).uniform_(1 - self.scale, 1 + self.scale).item()
        tx_frac = torch.empty(1).uniform_(0.5 - self.translate, 0.5 + self.translate).item()
        ty_frac = torch.empty(1).uniform_(0.5 - self.translate, 0.5 + self.translate).item()

        # Compute crop window size in input coords
        crop_w = round(out_w / s)
        crop_h = round(out_h / s)

        # Compute crop origin (center-biased)
        x_start = round(-tx_frac * out_w / s + in_w / 2)
        y_start = round(-ty_frac * out_h / s + in_h / 2)
        x_end = x_start + crop_w
        y_end = y_start + crop_h

        # Extract crop from input (with fill for out-of-bounds regions)
        crop = torch.full((image.shape[0], crop_h, crop_w), fill_val, dtype=image.dtype, device=image.device)

        src_x1 = max(x_start, 0)
        src_y1 = max(y_start, 0)
        src_x2 = min(x_end, in_w)
        src_y2 = min(y_end, in_h)

        dst_x1 = src_x1 - x_start
        dst_y1 = src_y1 - y_start
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            crop[:, dst_y1:dst_y2, dst_x1:dst_x2] = image[:, src_y1:src_y2, src_x1:src_x2]

        # Resize crop to output size
        if crop_h != out_h or crop_w != out_w:
            output = F.resize(crop, [out_h, out_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        else:
            output = crop

        # Effective scale factors from input crop to output
        effective_sx = out_w / crop_w
        effective_sy = out_h / crop_h

        # Transform bounding boxes
        if bboxes.numel() > 0:
            new_bboxes = bboxes.float().clone()
            new_bboxes[:, 0] = (bboxes[:, 0].float() - x_start) * effective_sx
            new_bboxes[:, 1] = (bboxes[:, 1].float() - y_start) * effective_sy
            new_bboxes[:, 2] = (bboxes[:, 2].float() - x_start) * effective_sx
            new_bboxes[:, 3] = (bboxes[:, 3].float() - y_start) * effective_sy

            # Clip to output bounds
            new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clamp(0, out_w)
            new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clamp(0, out_h)

            # Filter degenerate boxes
            new_w = new_bboxes[:, 2] - new_bboxes[:, 0]
            new_h = new_bboxes[:, 3] - new_bboxes[:, 1]

            orig_w_box = bboxes[:, 2].float() - bboxes[:, 0].float()
            orig_h_box = bboxes[:, 3].float() - bboxes[:, 1].float()
            scaled_orig_w = orig_w_box * s
            scaled_orig_h = orig_h_box * s

            eps = 1e-16
            area_ratio = (new_w * new_h) / (scaled_orig_w * scaled_orig_h + eps)
            ar = torch.maximum(new_w / (new_h + eps), new_h / (new_w + eps))
            valid = (new_w > 2) & (new_h > 2) & (area_ratio > self.area_thr) & (ar < 100)

            new_bboxes = new_bboxes[valid]
            labels = labels[valid]

            # Transform masks
            if masks is not None and masks.numel() > 0:
                mask_crop = torch.zeros(
                    (masks.shape[0], crop_h, crop_w),
                    dtype=masks.dtype,
                    device=masks.device,
                )
                if src_x2 > src_x1 and src_y2 > src_y1:
                    mask_crop[:, dst_y1:dst_y2, dst_x1:dst_x2] = masks[:, src_y1:src_y2, src_x1:src_x2]
                if crop_h != out_h or crop_w != out_w:
                    mask_crop = F.resize(
                        mask_crop,
                        [out_h, out_w],
                        interpolation=F.InterpolationMode.NEAREST,
                        antialias=False,
                    )
                masks = mask_crop[valid]
        else:
            new_bboxes = bboxes
            # No boxes to filter, but an empty masks tensor (0 instances) must still be
            # reshaped to the output canvas size: downstream transforms (e.g.
            # `CachedMixUp`) concatenate masks across samples and assume every sample's
            # masks tensor -- even an empty one -- matches (out_h, out_w), not the
            # pre-crop input size.
            if masks is not None:
                masks = torch.zeros((masks.shape[0], out_h, out_w), dtype=masks.dtype, device=masks.device)

        return output, new_bboxes, labels, masks

    @typing.no_type_check
    def forward(self, *_inputs: BaseSample) -> BaseSample:
        """Apply CachedMosaic with random perspective crop. Output is always img_scale."""
        if len(_inputs) != 1:
            msg = "Only single sample input is supported"
            raise ValueError(msg)
        inputs = _inputs[0]

        # Cache management (lightweight clone instead of deepcopy)
        cached_input = _clone_for_cache(inputs, max(1024, *self.img_scale))
        self._update_cache(cached_input)

        target_h, target_w = self.img_scale
        with_mask = hasattr(inputs, "masks") and inputs.masks is not None
        apply_mosaic = len(self.results_cache) >= 4 and torch.rand(1).item() <= self.prob

        if apply_mosaic:
            indices = self.get_indexes(self.results_cache)
            mix_results = [self.results_cache[i] for i in indices]
            mosaic_img, mosaic_bboxes, mosaic_labels, mosaic_masks = self._build_mosaic(
                cached_input,
                mix_results,
                with_mask,
            )
            out_img, out_bboxes, out_labels, out_masks = self._random_perspective_crop(
                mosaic_img,
                mosaic_bboxes,
                mosaic_labels,
                mosaic_masks,
                target_h,
                target_w,
            )
        else:
            # Non-mosaic path: letterbox resize + same random perspective
            img_lb, bboxes_lb, masks_lb = self._letterbox_resize(
                inputs.image,
                inputs.bboxes,
                inputs.masks if with_mask else None,
                target_h,
                target_w,
            )
            out_img, out_bboxes, out_labels, out_masks = self._random_perspective_crop(
                img_lb,
                bboxes_lb,
                inputs.label,
                masks_lb,
                target_h,
                target_w,
            )

        inputs.image = out_img.clamp(0, 1)
        inputs.bboxes = tv_tensors.BoundingBoxes(out_bboxes, format="XYXY", canvas_size=(target_h, target_w))
        inputs.label = out_labels
        if with_mask:
            inputs.masks = tv_tensors.Mask(
                out_masks
                if out_masks is not None
                else torch.zeros(
                    (0, target_h, target_w),
                    dtype=torch.uint8,
                    device=inputs.image.device,
                ),
            )
        inputs.img_info = _resized_crop_image_info(inputs.img_info, (target_h, target_w))
        return inputs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"img_scale={self.img_scale}, "
            f"center_ratio_range={self.center_ratio_range}, "
            f"pad_val={self.pad_val}, "
            f"prob={self.prob}, "
            f"max_cached_images={self.max_cached_images}, "
            f"random_pop={self.random_pop}, "
            f"scale={self.scale}, "
            f"translate={self.translate}, "
            f"area_thr={self.area_thr})"
        )


class CachedMixUp(CacheableMixin, tvt_v2.Transform):
    """Geometry-preserved MixUp augmentation for detection / instance segmentation.

    Blends the current image with a cached image using alpha blending without
    any geometric transformation.  Bounding boxes, labels and masks from both images are
    concatenated.

    If the cached image has a different spatial size it is resized to the current
    image size with ``keep_aspect_ratio`` padding so the aspect ratio is preserved.

    Args:
        img_scale (tuple[int, int]): Reference image size ``(H, W)`` used when
            the cached image needs to be rescaled.  Defaults to (640, 640).
        alpha (float): Alpha parameter for the ``Beta(alpha, alpha)`` distribution
            used to sample the blend ratio.  The mixed image is
            ``ratio * current + (1-ratio) * cached``.  Higher values concentrate
            samples around 0.5; lower values spread toward 0 and 1.
            Defaults to 1.5.
        pad_val (float): Padding fill value ([0, 255] or [0, 1]).
            Defaults to 114.0.
        max_iters (int): Max attempts to find a non-empty cached sample.
            Defaults to 15.
        bbox_clip_border (bool): Clip bboxes to image boundary after concat.
            Defaults to True.
        max_cached_images (int): Maximum cache size (>= 2). Defaults to 20.
        random_pop (bool): Random vs FIFO eviction. Defaults to True.
        p (float): Probability of applying MixUp. Defaults to 1.0.
    """

    def __init__(
        self,
        img_scale: tuple[int, int] | list[int] = (640, 640),  # (H, W)
        alpha: float = 1.5,
        pad_val: float = 114.0,
        max_iters: int = 15,
        bbox_clip_border: bool = True,
        max_cached_images: int = 20,
        random_pop: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__()

        if not isinstance(img_scale, (tuple, list)):
            msg = "img_scale must be a tuple or list"
            raise TypeError(msg)
        if max_cached_images < 2:
            msg = f"Cache size must be >= 2, got {max_cached_images}"
            raise ValueError(msg)
        if not 0 <= p <= 1.0:
            msg = f"Probability must be in [0,1], got {p}"
            raise ValueError(msg)
        if alpha <= 0:
            msg = f"alpha must be > 0 for Beta(alpha, alpha) sampling, got {alpha}"
            raise ValueError(msg)

        self.img_scale = tuple(img_scale)  # (H, W)
        self.alpha = alpha
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.prob = p
        self._init_cache()

    def _get_cached_index(self) -> int:
        """Return index of a cached sample with non-empty bboxes."""
        index = 0
        for _ in range(self.max_iters):
            index = int(torch.randint(0, len(self.results_cache), (1,)).item())
            if len(getattr(self.results_cache[index], "bboxes", [])) > 0:
                return index
        return index

    def _match_size(self, img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize *img* to ``(target_h, target_w)`` with keep-ratio + pad."""
        _, h, w = img.shape
        if h == target_h and w == target_w:
            return img
        scale = min(target_h / h, target_w / w)
        new_h, new_w = round(h * scale), round(w * scale)
        resized = F.resize(img, [new_h, new_w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        pad_val = self.pad_val / 255.0 if self.pad_val > 1.0 + 1e-5 else self.pad_val
        canvas = torch.full((img.shape[0], target_h, target_w), pad_val, dtype=img.dtype, device=img.device)
        canvas[:, :new_h, :new_w] = resized
        return canvas

    def _match_masks_size(
        self,
        masks: torch.Tensor,
        src_h: int,
        src_w: int,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Resize masks to ``(target_h, target_w)`` with keep-ratio + zero-pad."""
        if masks.numel() == 0:
            return torch.zeros((0, target_h, target_w), dtype=masks.dtype, device=masks.device)
        if src_h == target_h and src_w == target_w:
            return masks
        scale = min(target_h / src_h, target_w / src_w)
        new_h, new_w = round(src_h * scale), round(src_w * scale)
        resized = F.resize(masks, [new_h, new_w], interpolation=F.InterpolationMode.NEAREST, antialias=False)
        canvas = torch.zeros((masks.shape[0], target_h, target_w), dtype=masks.dtype, device=masks.device)
        canvas[:, :new_h, :new_w] = resized
        return canvas

    def _scale_bboxes(self, bboxes: torch.Tensor, src_h: int, src_w: int, dst_h: int, dst_w: int) -> torch.Tensor:
        """Scale XYXY bboxes from (src_h, src_w) to (dst_h, dst_w) keeping ratio + pad."""
        if bboxes.numel() == 0:
            return bboxes
        scale = min(dst_h / src_h, dst_w / src_w)
        return bboxes * scale

    @typing.no_type_check
    def forward(self, *_inputs: BaseSample) -> BaseSample:
        """Apply MixUp transform using pure torch operations."""
        if len(_inputs) != 1:
            msg = "Multiple inputs not supported"
            raise ValueError(msg)
        inputs = _inputs[0]

        # Cache management (lightweight clone instead of deepcopy)
        cached_input = _clone_for_cache(inputs, max(1024, *self.img_scale))
        self._update_cache(cached_input)

        # Early returns
        if len(self.results_cache) <= 1:
            return inputs

        if torch.rand(1).item() > self.prob:
            return inputs

        # Get cached sample (read-only: mixup creates new tensors via interpolation/concat,
        # never mutates cached.image/bboxes/label/masks in place)
        cache_idx = self._get_cached_index()
        cached = self.results_cache[cache_idx]

        if cached.bboxes.shape[0] == 0:
            return inputs

        if inputs.image.shape[-2:] != cached_input.image.shape[-2:]:
            Resize(size=cached_input.image.shape[-2:])(inputs)

        ori_img = inputs.image  # (C, H, W)  float [0,1]
        _, target_h, target_w = ori_img.shape
        cached_img = cached.image
        _, cached_h, cached_w = cached_img.shape

        # blend ratio (stochastic via Beta distribution)
        beta = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        # match cached image to current image size (geometry-preserved)
        cached_img = self._match_size(cached_img, target_h, target_w)

        # alpha blend
        mixup_img = (beta * ori_img + (1.0 - beta) * cached_img).clamp(0, 1)

        # rescale cached bboxes to the current canvas
        cached_bboxes = cached.bboxes.float()
        cached_bboxes = self._scale_bboxes(cached_bboxes, cached_h, cached_w, target_h, target_w)
        if self.bbox_clip_border and cached_bboxes.numel() > 0:
            cached_bboxes[..., 0::2] = cached_bboxes[..., 0::2].clamp(0, target_w)
            cached_bboxes[..., 1::2] = cached_bboxes[..., 1::2].clamp(0, target_h)

        # concatenate annotations
        mixup_bboxes = torch.cat([inputs.bboxes.float(), cached_bboxes], dim=0)
        mixup_labels = torch.cat([inputs.label, cached.label], dim=0)

        # filter degenerate boxes
        if mixup_bboxes.numel() > 0:
            w = mixup_bboxes[:, 2] - mixup_bboxes[:, 0]
            h = mixup_bboxes[:, 3] - mixup_bboxes[:, 1]
            valid = (w > 0) & (h > 0)
            mixup_bboxes = mixup_bboxes[valid]
            mixup_labels = mixup_labels[valid]
        else:
            valid = torch.zeros(0, dtype=torch.bool)

        # masks (instance segmentation)
        with_mask = hasattr(inputs, "masks") and inputs.masks is not None
        if with_mask:
            ori_masks = inputs.masks
            cached_masks = getattr(cached, "masks", None)
            if cached_masks is not None and len(cached_masks) > 0:
                cached_masks = self._match_masks_size(cached_masks, cached_h, cached_w, target_h, target_w)
                all_masks = torch.cat([ori_masks, cached_masks], dim=0)
                all_masks = all_masks[valid] if valid.numel() > 0 else all_masks
            else:
                all_valid = (
                    valid[: len(ori_masks)] if valid.numel() > 0 else torch.ones(len(ori_masks), dtype=torch.bool)
                )
                all_masks = ori_masks[all_valid]
            inputs.masks = tv_tensors.Mask(all_masks)

        # write back
        inputs.image = mixup_img
        inputs.bboxes = tv_tensors.BoundingBoxes(mixup_bboxes, format="XYXY", canvas_size=(target_h, target_w))
        inputs.label = mixup_labels

        return inputs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"img_scale={self.img_scale}, "
            f"alpha={self.alpha}, "
            f"pad_val={self.pad_val}, "
            f"max_cached_images={self.max_cached_images}, "
            f"random_pop={self.random_pop}, "
            f"prob={self.prob})"
        )


class FilterBoundingBoxes(tvt_v2.Transform):
    """Filter bounding boxes by size, aspect ratio, and area.

    Removes degenerate or severely distorted bounding boxes after geometric
    transforms (affine, crop, mosaic).  Matches the filtering behaviour of
    Ultralytics ``box_candidates`` without requiring access to pre-transform
    box dimensions.

    Args:
        min_wh: Minimum width AND height in pixels.  Boxes smaller than this
            in either dimension are removed.  Defaults to 2.
        max_aspect_ratio: Maximum aspect ratio ``max(w/h, h/w)``.  Boxes more
            extreme than this are removed.  Defaults to 20.
        min_area: Minimum box area in pixels (w * h).  Boxes smaller than this
            are removed.  Defaults to 4.
    """

    def __init__(
        self,
        min_wh: int = 2,
        max_aspect_ratio: float = 20.0,
        min_area: float = 4.0,
    ) -> None:
        super().__init__()
        self.min_wh = min_wh
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area = min_area

    @typing.no_type_check
    def forward(self, *_inputs: BaseSample) -> BaseSample:
        """Filter bounding boxes and corresponding labels/masks."""
        if len(_inputs) != 1:
            msg = f"Only single sample input is supported, got {len(_inputs)} samples."
            raise ValueError(msg)
        inputs = _inputs[0]

        bboxes = inputs.bboxes
        if bboxes is None or bboxes.numel() == 0:
            return inputs

        # Compute box dimensions (XYXY format)
        bboxes_f = bboxes.float()
        w = bboxes_f[:, 2] - bboxes_f[:, 0]
        h = bboxes_f[:, 3] - bboxes_f[:, 1]
        eps = 1e-6

        # Filter criteria (matching upstream box_candidates logic)
        valid_wh = (w >= self.min_wh) & (h >= self.min_wh)
        valid_area = (w * h) >= self.min_area
        aspect_ratio = torch.maximum(w / (h + eps), h / (w + eps))
        valid_ar = aspect_ratio < self.max_aspect_ratio

        valid = valid_wh & valid_area & valid_ar

        if valid.all():
            return inputs

        # Apply filter
        canvas_size = bboxes.canvas_size
        inputs.bboxes = tv_tensors.BoundingBoxes(
            bboxes[valid],
            format="XYXY",
            canvas_size=canvas_size,
        )
        inputs.label = inputs.label[valid]

        # Filter masks if present
        if hasattr(inputs, "masks") and inputs.masks is not None and len(inputs.masks) > 0:
            inputs.masks = tv_tensors.Mask(inputs.masks[valid])

        return inputs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"min_wh={self.min_wh}, "
            f"max_aspect_ratio={self.max_aspect_ratio}, "
            f"min_area={self.min_area})"
        )


class RandomIoUCrop(tvt_v2.RandomIoUCrop):
    """Random IoU crop with the option to set probability.

    Args:
        min_scale (float, optional): the same as RandomIoUCrop. Defaults to 0.3.
        max_scale (float, optional): the same as RandomIoUCrop. Defaults to 1.
        min_aspect_ratio (float, optional): the same as RandomIoUCrop. Defaults to 0.5.
        max_aspect_ratio (float, optional): the same as RandomIoUCrop. Defaults to 2.
        sampler_options (list[float] | None, optional): the same as RandomIoUCrop. Defaults to None.
        trials (int, optional): the same as RandomIoUCrop. Defaults to 40.
        p (float, optional): probability of applying the crop. Defaults to 1.0.
    """

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: list[float] | None = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale,
            max_scale,
            min_aspect_ratio,
            max_aspect_ratio,
            sampler_options,
            trials,
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:  # noqa: ANN401
        """Apply the transform to the given inputs."""
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().__call__(*inputs)


class MaskSafeRandomErasing(K.RandomErasing):
    """RandomErasing that erases images only, leaving masks unchanged.

    Kornia's RandomErasing overrides :meth:`apply_transform_mask` to also erase
    mask regions. This causes a shape mismatch in the GPU augmentation pipeline
    when used with instance segmentation because per-sample mask tensors have
    ``(N_instances, ...)`` dimensions while the inherited kornia params carry
    the full batch dimension ``(B, ...)``.

    Semantically, random erasing fills a rectangle on the image with a constant
    value — object boundaries (masks) are not changed.
    """

    def apply_transform_mask(
        self,
        input: torch.Tensor,  # noqa: A002  # matches parent class kornia signature
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Passthrough — masks are not affected by RandomErasing."""
        return input
