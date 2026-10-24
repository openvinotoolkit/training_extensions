# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Resize transform with aspect ratio preservation."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from torch import LongTensor
from torchvision import tv_tensors

from getitune.data.augmentation.transforms import Resize
from getitune.data.entity.sample import (
    DetectionSample,
    InstanceSegmentationSample,
)


class TestResize:
    """Test cases for Resize transform."""

    @pytest.fixture
    def square_image_entity(self) -> InstanceSegmentationSample:
        """Create a square image sample with bboxes and masks."""
        img_size = (100, 100)
        fake_image = torch.randint(low=0, high=256, size=(3, *img_size), dtype=torch.uint8)
        fake_bboxes = torch.tensor([[10, 10, 50, 50], [60, 20, 90, 80]], dtype=torch.float32)

        # Create masks that correspond to bboxes
        masks = torch.zeros(size=(2, *img_size), dtype=torch.uint8)
        masks[0, 10:50, 10:50] = 1
        masks[1, 20:80, 60:90] = 1

        return InstanceSegmentationSample(
            image=tv_tensors.Image(fake_image),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=tv_tensors.BoundingBoxes(  # type: ignore[call-overload]
                fake_bboxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=img_size
            ),
            label=LongTensor([0, 1]),
            masks=tv_tensors.Mask(masks),
        )

    @pytest.fixture
    def wide_image_entity(self) -> InstanceSegmentationSample:
        """Create a wide (landscape) image sample with bboxes and masks."""
        img_size = (100, 200)  # height, width
        fake_image = torch.randint(low=0, high=256, size=(3, *img_size), dtype=torch.uint8)
        fake_bboxes = torch.tensor([[10, 10, 50, 50], [120, 20, 180, 80]], dtype=torch.float32)

        masks = torch.zeros(size=(2, *img_size), dtype=torch.uint8)
        masks[0, 10:50, 10:50] = 1
        masks[1, 20:80, 120:180] = 1

        return InstanceSegmentationSample(
            image=tv_tensors.Image(fake_image),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=tv_tensors.BoundingBoxes(  # type: ignore[call-overload]
                fake_bboxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=img_size
            ),
            label=LongTensor([0, 1]),
            masks=tv_tensors.Mask(masks),
        )

    @pytest.fixture
    def tall_image_entity(self) -> InstanceSegmentationSample:
        """Create a tall (portrait) image sample with bboxes and masks."""
        img_size = (200, 100)  # height, width
        fake_image = torch.randint(low=0, high=256, size=(3, *img_size), dtype=torch.uint8)
        fake_bboxes = torch.tensor([[10, 10, 50, 50], [60, 20, 90, 180]], dtype=torch.float32)

        masks = torch.zeros(size=(2, *img_size), dtype=torch.uint8)
        masks[0, 10:50, 10:50] = 1
        masks[1, 20:180, 60:90] = 1

        return InstanceSegmentationSample(
            image=tv_tensors.Image(fake_image),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=tv_tensors.BoundingBoxes(  # type: ignore[call-overload]
                fake_bboxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=img_size
            ),
            label=LongTensor([0, 1]),
            masks=tv_tensors.Mask(masks),
        )

    # ==================== Standard Resize Tests ====================

    def test_resize_square_to_square(self, square_image_entity: InstanceSegmentationSample) -> None:
        """Test resizing square image to square target without aspect ratio preservation."""
        resize = Resize(size=(64, 64), resize_targets=True, keep_aspect_ratio=False)
        entity = deepcopy(square_image_entity)
        orig_bboxes = entity.bboxes.clone()
        orig_h, orig_w = entity.image.shape[-2:]

        result = resize(entity)

        assert result.image.shape[-2:] == (64, 64)
        assert result.masks.shape[-2:] == (64, 64)
        # Bboxes should be scaled proportionally
        scale_x = 64 / orig_w
        scale_y = 64 / orig_h
        expected_bboxes = orig_bboxes.clone()
        expected_bboxes[:, 0::2] = orig_bboxes[:, 0::2] * scale_x
        expected_bboxes[:, 1::2] = orig_bboxes[:, 1::2] * scale_y
        assert torch.allclose(result.bboxes.float(), expected_bboxes.float(), atol=1.0)

    def test_resize_wide_to_square(self, wide_image_entity: InstanceSegmentationSample) -> None:
        """Test resizing wide image to square target without aspect ratio preservation."""
        resize = Resize(size=(64, 64), resize_targets=True, keep_aspect_ratio=False)
        entity = deepcopy(wide_image_entity)

        result = resize(entity)

        assert result.image.shape[-2:] == (64, 64)
        assert result.masks.shape[-2:] == (64, 64)

    def test_resize_targets_false(self, square_image_entity: InstanceSegmentationSample) -> None:
        """Test that resize_targets=False only resizes image."""
        resize = Resize(size=(64, 64), resize_targets=False, keep_aspect_ratio=False)
        entity = deepcopy(square_image_entity)
        original_bboxes = entity.bboxes.clone()
        assert entity.masks is not None
        original_masks_shape = entity.masks.shape[-2:]

        result = resize(entity)

        assert result.image.shape[-2:] == (64, 64)
        # Bboxes and masks should be unchanged
        assert torch.equal(result.bboxes, original_bboxes)
        assert result.masks is not None
        assert result.masks.shape[-2:] == original_masks_shape

    # ==================== Aspect Ratio Preservation Tests ====================

    def test_resize_with_aspect_ratio_square_to_square(self, square_image_entity: InstanceSegmentationSample) -> None:
        """Test aspect ratio resize of square image to square target (no padding needed)."""
        resize = Resize(size=(64, 64), resize_targets=True, keep_aspect_ratio=True)
        entity = deepcopy(square_image_entity)
        orig_bboxes = entity.bboxes.clone()
        orig_h, orig_w = entity.image.shape[-2:]

        result = resize(entity)

        # Square to square: no padding needed
        assert result.image.shape[-2:] == (64, 64)
        assert result.masks.shape[-2:] == (64, 64)
        # Scale is uniform for square to square
        scale = min(64 / orig_w, 64 / orig_h)
        expected_bboxes = orig_bboxes * scale
        assert torch.allclose(result.bboxes.float(), expected_bboxes.float(), atol=1.0)

    def test_resize_with_aspect_ratio_wide_to_square(self, wide_image_entity: InstanceSegmentationSample) -> None:
        """Test aspect ratio resize of wide image to square target (vertical padding)."""
        resize = Resize(size=(128, 128), resize_targets=True, keep_aspect_ratio=True)
        entity = deepcopy(wide_image_entity)
        orig_bboxes = entity.bboxes.clone()
        orig_h, orig_w = entity.image.shape[-2:]  # 100, 200

        result = resize(entity)

        # Output should be exactly target size
        assert result.image.shape[-2:] == (128, 128)
        assert result.masks.shape[-2:] == (128, 128)

        # Wide image (200w x 100h) -> scale by min(128/200, 128/100) = 0.64
        # Resized: 128w x 64h, then pad bottom-right only (pad_bottom=64)
        scale = min(128 / orig_w, 128 / orig_h)
        new_h = round(orig_h * scale)  # 64
        pad_bottom = 128 - new_h  # 64

        # Check that padding info is stored (pad_left, pad_top, pad_right, pad_bottom)
        assert hasattr(result.img_info, "pad_offset")
        assert result.img_info.pad_offset[0] == 0  # pad_left
        assert result.img_info.pad_offset[1] == 0  # pad_top
        assert result.img_info.pad_offset[3] == pad_bottom  # pad_bottom

        # Verify bboxes are correctly transformed (scale only, no offset since pad is bottom-right)
        expected_x1 = orig_bboxes[:, 0] * scale
        expected_y1 = orig_bboxes[:, 1] * scale
        assert torch.allclose(result.bboxes[:, 0].float(), expected_x1.float(), atol=1.0)
        assert torch.allclose(result.bboxes[:, 1].float(), expected_y1.float(), atol=1.0)

    def test_resize_with_aspect_ratio_tall_to_square(self, tall_image_entity: InstanceSegmentationSample) -> None:
        """Test aspect ratio resize of tall image to square target (horizontal padding)."""
        resize = Resize(size=(128, 128), resize_targets=True, keep_aspect_ratio=True)
        entity = deepcopy(tall_image_entity)
        orig_bboxes = entity.bboxes.clone()
        orig_h, orig_w = entity.image.shape[-2:]  # 200, 100

        result = resize(entity)

        # Output should be exactly target size
        assert result.image.shape[-2:] == (128, 128)
        assert result.masks.shape[-2:] == (128, 128)

        # Tall image (100w x 200h) -> scale by min(128/100, 128/200) = 0.64
        # Resized: 64w x 128h, then pad bottom-right only (pad_right=64)
        scale = min(128 / orig_w, 128 / orig_h)
        new_w = round(orig_w * scale)  # 64
        pad_right = 128 - new_w  # 64

        # Check that padding info is stored (pad_left, pad_top, pad_right, pad_bottom)
        assert hasattr(result.img_info, "pad_offset")
        assert result.img_info.pad_offset[0] == 0  # pad_left
        assert result.img_info.pad_offset[1] == 0  # pad_top
        assert result.img_info.pad_offset[2] == pad_right  # pad_right

        # Verify bboxes are correctly transformed (scale only, no offset since pad is bottom-right)
        expected_x1 = orig_bboxes[:, 0] * scale
        expected_y1 = orig_bboxes[:, 1] * scale
        assert torch.allclose(result.bboxes[:, 0].float(), expected_x1.float(), atol=1.0)
        assert torch.allclose(result.bboxes[:, 1].float(), expected_y1.float(), atol=1.0)

    def test_resize_with_aspect_ratio_to_non_square(self, wide_image_entity: InstanceSegmentationSample) -> None:
        """Test aspect ratio resize to non-square target."""
        resize = Resize(size=(96, 128), resize_targets=True, keep_aspect_ratio=True)  # h, w
        entity = deepcopy(wide_image_entity)
        orig_h, orig_w = entity.image.shape[-2:]  # 100, 200

        result = resize(entity)

        # Output should be exactly target size
        assert result.image.shape[-2:] == (96, 128)
        assert result.masks.shape[-2:] == (96, 128)

    def test_resize_pad_value(self, wide_image_entity: InstanceSegmentationSample) -> None:
        """Test that pad_value is correctly applied."""
        pad_value = 128
        resize = Resize(size=(128, 128), keep_aspect_ratio=True, pad_value=pad_value)
        entity = deepcopy(wide_image_entity)

        result = resize(entity)

        # Check that padding areas have the correct value
        # For wide image, padding is on top and bottom
        # The top row should be padded (if pad_top > 0)
        pad_top = result.img_info.pad_offset[1]
        if pad_top > 0:
            top_row_mean = result.image[:, 0, :].float().mean()
            assert abs(top_row_mean - pad_value) < 1.0

    def test_resize_pad_value_normalised_for_float_image(self, wide_image_entity: InstanceSegmentationSample) -> None:
        """``pad_value`` expressed in 0-255 range must be rescaled for float images.

        Recipes (e.g. RTMDet/YOLOX letterbox) configure ``pad_value: 114`` while
        the runtime image tensor is float32 in ``[0, 1]``. Without rescaling, the
        padded region would be filled with 114.0, completely dominating any
        downstream ImageNet-style normalisation.
        """
        pad_value = 114
        resize = Resize(
            size=(128, 128),
            keep_aspect_ratio=True,
            resize_targets=False,
            pad_value=pad_value,
        )
        entity = deepcopy(wide_image_entity)
        # Mimic the production CPU pipeline which scales uint8 → float32 in [0, 1]
        # before any geometric augmentation runs.
        entity.image = tv_tensors.Image(entity.image.float().div(255.0))

        result = resize(entity)

        # Wide image (100x200) → resized to 128x64 → padded bottom-right.
        # The bottom row should be entirely padding.
        pad_bottom = result.img_info.pad_offset[3]
        assert pad_bottom > 0
        bottom_row_mean = result.image[:, -1, :].float().mean().item()
        assert abs(bottom_row_mean - pad_value / 255.0) < 1e-3
        # Padding must stay within the image's [0, 1] range.
        assert result.image.max().item() <= 1.0 + 1e-5

    def test_resize_masks_binary_preserved(self, square_image_entity: InstanceSegmentationSample) -> None:
        """Test that mask binary values are preserved after resize."""
        resize = Resize(size=(64, 64), resize_targets=True, keep_aspect_ratio=True)
        entity = deepcopy(square_image_entity)

        result = resize(entity)

        # Masks should only contain 0s and 1s (or near that for interpolation)
        unique_values = torch.unique(result.masks)
        assert all(v in {0, 1} for v in unique_values.tolist())

    # ==================== Edge Cases ====================

    def test_resize_empty_bboxes(self) -> None:
        """Test resize with no bounding boxes."""
        img_size = (100, 100)
        entity = DetectionSample(
            image=tv_tensors.Image(torch.randint(0, 256, (3, *img_size), dtype=torch.uint8)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=tv_tensors.BoundingBoxes(  # type: ignore[call-overload]
                torch.empty((0, 4), dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_size,
            ),
            label=LongTensor([]),
        )
        resize = Resize(size=(64, 64), keep_aspect_ratio=True)

        result = resize(entity)

        assert result.image.shape[-2:] == (64, 64)
        assert len(result.bboxes) == 0

    def test_resize_empty_masks(self) -> None:
        """Test resize with empty masks preserves spatial dimensions matching the image."""
        img_size = (100, 100)
        entity = InstanceSegmentationSample(
            image=tv_tensors.Image(torch.randint(0, 256, (3, *img_size), dtype=torch.uint8)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=tv_tensors.BoundingBoxes(  # type: ignore[call-overload]
                torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_size,
            ),
            label=LongTensor([0]),
            masks=tv_tensors.Mask(torch.empty((0, *img_size), dtype=torch.uint8)),
        )
        resize = Resize(size=(64, 64), keep_aspect_ratio=True)

        result = resize(entity)

        assert result.image.shape[-2:] == (64, 64)
        assert result.masks.shape[0] == 0
        # Spatial dimensions of empty masks must match the resized/padded image
        assert result.masks.shape[-2:] == result.image.shape[-2:]

    def test_resize_empty_masks_non_square(self) -> None:
        """Test resize empty masks with non-square image (padding required)."""
        img_size = (100, 200)  # wide image
        entity = InstanceSegmentationSample(
            image=tv_tensors.Image(torch.randint(0, 256, (3, *img_size), dtype=torch.uint8)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=tv_tensors.BoundingBoxes(  # type: ignore[call-overload]
                torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_size,
            ),
            label=LongTensor([0]),
            masks=tv_tensors.Mask(torch.empty((0, *img_size), dtype=torch.uint8)),
        )
        resize = Resize(size=(128, 128), keep_aspect_ratio=True)

        result = resize(entity)

        assert result.image.shape[-2:] == (128, 128)
        assert result.masks.shape[0] == 0
        assert result.masks.shape[-2:] == result.image.shape[-2:]

    def test_resize_single_int_size(self) -> None:
        """Test that single int size is converted to tuple."""
        resize = Resize(size=64, keep_aspect_ratio=True)
        assert resize.size == (64, 64)

    def test_resize_tensor_directly(self) -> None:
        """Test resizing a tensor directly (fallback path)."""
        tensor = torch.randint(0, 256, (3, 100, 100), dtype=torch.uint8)
        resize = Resize(size=(64, 64), keep_aspect_ratio=False)

        result = resize(tensor)

        assert result.shape[-2:] == (64, 64)

    # ==================== Consistency Tests ====================

    def test_bbox_inside_image_after_resize(self, square_image_entity: InstanceSegmentationSample) -> None:
        """Test that all bboxes remain inside image bounds after resize."""
        resize = Resize(size=(64, 64), resize_targets=True, keep_aspect_ratio=True)
        entity = deepcopy(square_image_entity)

        result = resize(entity)

        h, w = result.image.shape[-2:]
        # All bbox coordinates should be within [0, w] for x and [0, h] for y
        assert torch.all(result.bboxes[:, 0] >= 0)
        assert torch.all(result.bboxes[:, 1] >= 0)
        assert torch.all(result.bboxes[:, 2] <= w)
        assert torch.all(result.bboxes[:, 3] <= h)

    def test_mask_same_size_as_image(self, square_image_entity: InstanceSegmentationSample) -> None:
        """Test that masks have same spatial size as image after resize."""
        resize = Resize(size=(64, 64), resize_targets=True, keep_aspect_ratio=True)
        entity = deepcopy(square_image_entity)

        result = resize(entity)

        assert result.masks.shape[-2:] == result.image.shape[-2:]

    def test_img_info_updated(self, square_image_entity: InstanceSegmentationSample) -> None:
        """Test that img_info is correctly updated after resize."""
        resize = Resize(size=(64, 64), resize_targets=True, keep_aspect_ratio=True)
        entity = deepcopy(square_image_entity)

        result = resize(entity)

        assert result.img_info.img_shape == (64, 64)

    def test_scale_factor_stored(self, wide_image_entity: InstanceSegmentationSample) -> None:
        """Test that scale factor is stored in img_info when using aspect ratio mode."""
        resize = Resize(size=(128, 128), resize_targets=True, keep_aspect_ratio=True)
        entity = deepcopy(wide_image_entity)
        orig_h, orig_w = entity.image.shape[-2:]

        result = resize(entity)

        # Scale factor should be stored
        assert hasattr(result.img_info, "scale_factor")
        expected_scale = min(128 / orig_w, 128 / orig_h)
        assert abs(result.img_info.scale_factor[0] - expected_scale) < 0.01
        assert abs(result.img_info.scale_factor[1] - expected_scale) < 0.01

    # ==================== Early Exit Tests ====================

    def test_early_exit_same_size(self, square_image_entity: InstanceSegmentationSample) -> None:
        """Test that Resize returns input unchanged when image is already at target size."""
        entity = deepcopy(square_image_entity)
        # Image is 100x100, resize target is 100x100 -> early exit
        resize = Resize(size=(100, 100), resize_targets=True, keep_aspect_ratio=True)
        result = resize(entity)
        # Should return the exact same object (no copy)
        assert result is entity
        assert result.image.shape[-2:] == (100, 100)
