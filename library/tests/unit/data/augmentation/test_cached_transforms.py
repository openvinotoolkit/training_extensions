# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CachedMosaic, CachedMixUp, RandomIoUCrop, and _clone_for_cache."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from torchvision import tv_tensors

from getitune.data.augmentation.cache import _clone_for_cache
from getitune.data.augmentation.transforms import CachedMixUp, CachedMosaic, RandomIoUCrop
from getitune.data.entity.sample import DetectionSample, InstanceSegmentationSample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_det_sample(
    h: int = 64,
    w: int = 64,
    n_boxes: int = 2,
) -> InstanceSegmentationSample:
    """Create a minimal instance-seg sample for testing (always has masks)."""
    image = torch.rand(3, h, w)
    bboxes = torch.zeros(n_boxes, 4, dtype=torch.float32)
    for i in range(n_boxes):
        x1, y1 = 5 + i * 10, 5 + i * 10
        bboxes[i] = torch.tensor([x1, y1, x1 + 20, y1 + 20])
    labels = torch.arange(n_boxes, dtype=torch.long)
    masks_t = torch.zeros(n_boxes, h, w, dtype=torch.uint8)
    for i in range(n_boxes):
        x1, y1 = int(bboxes[i, 0]), int(bboxes[i, 1])
        x2, y2 = int(bboxes[i, 2]), int(bboxes[i, 3])
        masks_t[i, y1:y2, x1:x2] = 1

    return InstanceSegmentationSample(
        image=tv_tensors.Image(image),
        dm_image_info=DmImageInfo(height=h, width=w),
        bboxes=tv_tensors.BoundingBoxes(bboxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)),  # type: ignore[no-matching-overload]
        label=labels,
        masks=tv_tensors.Mask(masks_t),
    )


# =====================================================================
# CachedMosaic Tests
# =====================================================================
class TestCachedMosaicInit:
    """Validation tests for CachedMosaic.__init__."""

    def test_invalid_img_scale_type_raises(self):
        with pytest.raises(TypeError, match="img_scale must be a tuple or list"):
            CachedMosaic(img_scale=640)  # type: ignore[arg-type]

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError, match="probability must be in"):
            CachedMosaic(p=1.5)

    def test_invalid_max_cached_images_raises(self):
        with pytest.raises(ValueError, match="max_cached_images must be >= 4"):
            CachedMosaic(max_cached_images=3)

    def test_valid_construction(self):
        m = CachedMosaic(img_scale=(320, 320), p=0.5, max_cached_images=10)
        assert m.img_scale == (320, 320)
        assert m.prob == 0.5
        assert m.max_cached_images == 10


class TestCachedMosaicForward:
    """Functional tests for CachedMosaic augmentation."""

    def test_oversized_image_is_reduced_before_caching(self):
        mosaic = CachedMosaic(img_scale=(640, 640), p=0.0)
        sample = _make_det_sample(h=1200, w=800)
        original_boxes = sample.bboxes.clone()
        original_pixels = sample.image.numel()

        mosaic(sample)

        cached = mosaic.results_cache[0]
        assert cached.image.shape[-2:] == (1024, 683)
        assert cached.image.numel() < original_pixels
        assert cached.masks is not None
        assert cached.masks.shape[-2:] == (1024, 683)
        assert cached.masks[0, 10, 10] == 1
        assert cached.masks[0, 50, 50] == 0
        assert not isinstance(cached.bboxes, tv_tensors.BoundingBoxes)
        assert torch.allclose(cached.bboxes[:, 0::2], original_boxes[:, 0::2] * (683 / 800))
        assert torch.allclose(cached.bboxes[:, 1::2], original_boxes[:, 1::2] * (1024 / 1200))

    def test_cache_too_small_returns_input(self):
        """With fewer than 4 cached images, forward returns img_scale (non-mosaic path)."""
        mosaic = CachedMosaic(img_scale=(32, 32), p=1.0, max_cached_images=10)
        sample = _make_det_sample(h=32, w=32)
        result = mosaic(sample)
        # After first call, cache has 1 item → non-mosaic path → output is img_scale
        assert result.image.shape[-2:] == (32, 32)

    def test_mosaic_applied_after_cache_fills(self):
        """After 4+ samples cached, mosaic produces img_scale output (with perspective crop)."""
        mosaic = CachedMosaic(img_scale=(32, 32), p=1.0, max_cached_images=40)
        for _ in range(4):
            sample = _make_det_sample(h=32, w=32, n_boxes=2)
            result = mosaic(sample)
        # After 4 calls, cache is full enough; 5th call should produce mosaic
        sample = _make_det_sample(h=32, w=32, n_boxes=2)
        result = mosaic(sample)
        # New behavior: mosaic output is always img_scale (perspective crop applied)
        assert result.image.shape[-2:] == (32, 32)
        # Image values should be in [0, 1]
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_mosaic_assembles_reduced_cached_tiles(self):
        mosaic = CachedMosaic(img_scale=(640, 640), p=1.0, max_cached_images=4)
        build_mosaic = mosaic._build_mosaic
        tile_shapes = []

        def capture_tile_shape(
            inputs, mix_results, with_mask
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
            tile_shapes.append(tuple(inputs.image.shape[-2:]))
            return build_mosaic(inputs, mix_results, with_mask)

        with patch.object(mosaic, "_build_mosaic", side_effect=capture_tile_shape):
            for _ in range(4):
                result = mosaic(_make_det_sample(h=1100, w=550, n_boxes=0))

        assert tile_shapes == [(1024, 512)]
        assert all(cached.image.shape[-2:] == (1024, 512) for cached in mosaic.results_cache)
        assert result.image.shape[-2:] == (640, 640)
        assert result.masks is not None
        assert result.masks.shape[-2:] == (640, 640)

    def test_mosaic_with_masks(self):
        """Mosaic should handle instance segmentation masks."""
        mosaic = CachedMosaic(img_scale=(32, 32), p=1.0, max_cached_images=40)
        for _ in range(5):
            sample = _make_det_sample(h=32, w=32, n_boxes=2)
            result = mosaic(sample)
        # After enough samples, mosaic should produce masks at img_scale
        assert result.masks is not None
        assert result.masks.shape[-2:] == (32, 32)

    def test_probability_zero_returns_input(self):
        """With probability=0, mosaic never applies (even with full cache)."""
        mosaic = CachedMosaic(img_scale=(32, 32), p=0.0, max_cached_images=40)
        for _ in range(5):
            sample = _make_det_sample(h=32, w=32)
            _ = mosaic(sample)
        sample = _make_det_sample(h=32, w=32)
        result = mosaic(sample)
        # prob=0 → always skip → original size preserved
        assert result.image.shape[-2:] == (32, 32)

    def test_cache_eviction(self):
        """Cache should not grow beyond max_cached_images."""
        mosaic = CachedMosaic(img_scale=(32, 32), p=0.0, max_cached_images=5)
        for _ in range(10):
            sample = _make_det_sample(h=32, w=32)
            mosaic(sample)
        assert len(mosaic.results_cache) <= 5

    def test_bboxes_valid_after_mosaic(self):
        """Bboxes should be valid XYXY after mosaic (x2 > x1, y2 > y1)."""
        mosaic = CachedMosaic(img_scale=(32, 32), p=1.0, max_cached_images=40)
        for _ in range(5):
            sample = _make_det_sample(h=32, w=32, n_boxes=3)
            result = mosaic(sample)
        # After mosaic, bboxes should be valid
        if len(result.bboxes) > 0:
            widths = result.bboxes[:, 2] - result.bboxes[:, 0]
            heights = result.bboxes[:, 3] - result.bboxes[:, 1]
            assert (widths > 0).all()
            assert (heights > 0).all()

    def test_labels_match_bboxes(self):
        """Labels should be aligned with bboxes after mosaic."""
        mosaic = CachedMosaic(img_scale=(32, 32), p=1.0, max_cached_images=40)
        for _ in range(5):
            sample = _make_det_sample(h=32, w=32, n_boxes=2)
            result = mosaic(sample)
        assert result.bboxes.shape[0] == result.label.shape[0]

    def test_get_indexes(self):
        """get_indexes should return 3 random indices within cache bounds."""
        mosaic = CachedMosaic(img_scale=(32, 32))
        cache = list(range(10))
        indices = mosaic.get_indexes(cache)
        assert len(indices) == 3
        assert all(0 <= i < 10 for i in indices)

    def test_repr(self):
        mosaic = CachedMosaic(img_scale=(320, 320), p=0.8)
        r = repr(mosaic)
        assert "CachedMosaic" in r
        assert "320" in r
        assert "0.8" in r


# =====================================================================
# CachedMixUp Tests
# =====================================================================
class TestCachedMixUpInit:
    """Validation tests for CachedMixUp.__init__."""

    def test_invalid_img_scale_type_raises(self):
        with pytest.raises(TypeError, match="img_scale must be a tuple or list"):
            CachedMixUp(img_scale=640)  # type: ignore[arg-type]

    def test_invalid_max_cached_images_raises(self):
        with pytest.raises(ValueError, match="Cache size must be >= 2"):
            CachedMixUp(max_cached_images=1)

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError, match="Probability must be in"):
            CachedMixUp(p=-0.1)

    def test_invalid_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            CachedMixUp(alpha=0.0)

    def test_invalid_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            CachedMixUp(alpha=-1.0)

    def test_valid_construction(self):
        m = CachedMixUp(img_scale=(320, 320), p=0.5, max_cached_images=10)
        assert m.img_scale == (320, 320)
        assert m.prob == 0.5


class TestCachedMixUpForward:
    """Functional tests for CachedMixUp augmentation."""

    @pytest.mark.parametrize(
        ("img_scale", "height", "width", "cached_shape"),
        [
            ((640, 640), 1200, 800, (1024, 683)),
            ((1280, 1280), 1200, 800, (1200, 800)),
            ((1280, 1280), 1400, 700, (1280, 640)),
        ],
    )
    def test_cache_limit_respects_model_resolution(self, img_scale, height, width, cached_shape):
        mixup = CachedMixUp(img_scale=img_scale, p=0.0)
        sample = _make_det_sample(h=height, w=width, n_boxes=0)

        result = mixup(sample)

        cached = mixup.results_cache[0]
        assert cached.image.shape[-2:] == cached_shape
        assert cached.masks is not None
        assert cached.masks.shape == (0, *cached_shape)
        assert cached.bboxes.shape == (0, 4)
        assert result.image.shape[-2:] == (height, width)

    def test_cache_too_small_returns_input(self):
        """With only 1 cached sample, forward returns input unchanged."""
        mixup = CachedMixUp(img_scale=(32, 32), p=1.0, max_cached_images=20)
        sample = _make_det_sample(h=32, w=32)
        result = mixup(sample)
        # First call: cache has 1 item → early return
        assert result.image.shape == sample.image.shape

    def test_mixup_applied_after_cache_fills(self):
        """After 2+ samples cached, mixup blends images."""
        mixup = CachedMixUp(
            img_scale=(32, 32),
            p=1.0,
            max_cached_images=20,
        )
        for _ in range(3):
            sample = _make_det_sample(h=32, w=32, n_boxes=2)
            result = mixup(sample)
        # After 3 calls, cache has enough, mixup should produce valid result
        assert result.image.shape[-2:] == (32, 32)
        # Combined bboxes: at least original count
        assert result.bboxes.shape[0] >= 2

    @pytest.mark.parametrize(
        ("img_scale", "height", "width", "cached_shape"),
        [
            ((640, 640), 1100, 550, (1024, 512)),
            ((1280, 1280), 1200, 800, (1200, 800)),
            ((1280, 1280), 1400, 700, (1280, 640)),
        ],
    )
    def test_mixup_blends_reduced_cached_images(self, img_scale, height, width, cached_shape):
        mixup = CachedMixUp(img_scale=img_scale, p=1.0)
        mixup(_make_det_sample(h=height, w=width, n_boxes=1))
        result = mixup(_make_det_sample(h=height, w=width, n_boxes=1))

        assert result.image.shape[-2:] == cached_shape
        assert result.bboxes.shape[0] == result.label.shape[0] == 2
        assert result.masks is not None
        assert result.masks.shape == (2, *cached_shape)
        assert all(cached.image.shape[-2:] == cached_shape for cached in mixup.results_cache)

    def test_probability_zero_returns_input(self):
        """With probability=0, mixup never applies."""
        mixup = CachedMixUp(img_scale=(32, 32), p=0.0, max_cached_images=20)
        for _ in range(5):
            sample = _make_det_sample(h=32, w=32, n_boxes=2)
            result = mixup(sample)
        # prob=0 → always skip → bboxes should be exactly the input count
        assert result.bboxes.shape[0] == 2

    def test_cache_eviction(self):
        """Cache should not grow beyond max_cached_images."""
        mixup = CachedMixUp(img_scale=(32, 32), p=0.0, max_cached_images=5)
        for _ in range(10):
            mixup(_make_det_sample(h=32, w=32))
        assert len(mixup.results_cache) <= 5

    def test_labels_match_bboxes(self):
        """Labels count must match bboxes count after mixup."""
        mixup = CachedMixUp(img_scale=(32, 32), p=1.0, max_cached_images=20)
        for _ in range(5):
            result = mixup(_make_det_sample(h=32, w=32, n_boxes=2))
        assert result.bboxes.shape[0] == result.label.shape[0]

    def test_mixup_with_masks(self):
        """MixUp should handle instance segmentation masks."""
        mixup = CachedMixUp(img_scale=(32, 32), p=1.0, max_cached_images=20)
        for _ in range(5):
            result = mixup(_make_det_sample(h=32, w=32, n_boxes=2))
        assert result.masks is not None
        # Masks count should match bboxes count
        assert result.masks.shape[0] == result.bboxes.shape[0]

    def test_image_clamped_to_unit(self):
        """MixUp output image should be clamped to [0, 1]."""
        mixup = CachedMixUp(img_scale=(32, 32), p=1.0, max_cached_images=20)
        for _ in range(5):
            result = mixup(_make_det_sample(h=32, w=32))
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_repr(self):
        m = CachedMixUp(img_scale=(640, 640), alpha=2.0)
        r = repr(m)
        assert "CachedMixUp" in r
        assert "640" in r
        assert "2.0" in r


# =====================================================================
# RandomIoUCrop Tests
# =====================================================================
class TestRandomIoUCrop:
    """Tests for RandomIoUCrop probability gating."""

    def test_probability_zero_passthrough(self):
        """With p=0, input is returned unchanged."""
        crop = RandomIoUCrop(p=0.0)
        image = tv_tensors.Image(torch.rand(3, 100, 100))
        bboxes = tv_tensors.BoundingBoxes(  # type: ignore[no-matching-overload]
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(100, 100),
        )
        labels = torch.tensor([0])
        result = crop(image, bboxes, labels)
        # p=0 → always skip → returns tuple of inputs
        assert isinstance(result, tuple)
        assert torch.equal(result[0], image)

    def test_probability_one_applies(self):
        """With p=1, crop always applies (output shape may differ)."""
        crop = RandomIoUCrop(p=1.0)
        image = tv_tensors.Image(torch.rand(3, 100, 100))
        bboxes = tv_tensors.BoundingBoxes(  # type: ignore[no-matching-overload]
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(100, 100),
        )
        labels = torch.tensor([0])
        # Should not raise
        result = crop(image, bboxes, labels)
        assert result is not None

    def test_single_input_returns_single(self):
        """With p=0 and single input, returns the input (not a tuple)."""
        crop = RandomIoUCrop(p=0.0)
        image = tv_tensors.Image(torch.rand(3, 50, 50))
        result = crop(image)
        # Single input + skip → returns single tensor
        assert isinstance(result, torch.Tensor)


# =====================================================================
# _clone_for_cache Tests
# =====================================================================
class TestCloneForCache:
    """Tests for _clone_for_cache utility function."""

    def test_clone_detection_sample(self):
        """Clone a DetectionSample (no masks) into _CachedSample."""
        sample = DetectionSample(
            image=tv_tensors.Image(torch.rand(3, 64, 64)),
            dm_image_info=DmImageInfo(height=64, width=64),
            bboxes=tv_tensors.BoundingBoxes(  # type: ignore[no-matching-overload]
                torch.tensor([[5, 5, 25, 25], [30, 30, 55, 55]], dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(64, 64),
            ),
            label=torch.tensor([0, 1], dtype=torch.long),
        )
        cached = _clone_for_cache(sample)

        assert torch.equal(cached.image, sample.image)
        assert torch.equal(cached.bboxes, sample.bboxes)
        assert sample.label is not None
        assert torch.equal(cached.label, sample.label)
        assert cached.masks is None
        # Verify it's a clone (not the same tensor)
        assert cached.image.data_ptr() != sample.image.data_ptr()

    def test_clone_instance_segmentation_sample(self):
        """Clone an InstanceSegmentationSample (with masks) into _CachedSample."""
        sample = _make_det_sample(h=64, w=64, n_boxes=3)
        cached = _clone_for_cache(sample)

        assert torch.equal(cached.image, sample.image)
        assert torch.equal(cached.bboxes, sample.bboxes)
        assert sample.label is not None
        assert torch.equal(cached.label, sample.label)
        assert cached.masks is not None
        assert sample.masks is not None
        assert torch.equal(cached.masks, sample.masks)
        # Verify cloned (not the same tensor)
        assert cached.image.data_ptr() != sample.image.data_ptr()
        assert cached.masks.data_ptr() != sample.masks.data_ptr()
