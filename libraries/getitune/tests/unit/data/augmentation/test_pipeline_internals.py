# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _IntensityAdapter, _sanitize_annotations, and GPU pipeline edge cases."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from getitune.data.augmentation.pipeline import (
    _DTYPE_TO_BIT_DEPTH,
    GPUAugmentationPipeline,
    _IntensityAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _IdentityTransform(nn.Module):
    """No-op transform for testing _IntensityAdapter wrapping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ScaleTransform(nn.Module):
    """Multiply by 0.5 for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5


class _SimpleSample:
    """Minimal duck-type of BaseSample for testing."""

    def __init__(self, image: torch.Tensor, img_info: object | None = None) -> None:
        self.image = image
        self.img_info = img_info


class _SimpleImgInfo:
    """Minimal duck-type of ImageInfo for testing."""

    def __init__(self) -> None:
        self.bit_depth: int | None = None


# =====================================================================
# _DTYPE_TO_BIT_DEPTH mapping
# =====================================================================
class TestDtypeToBitDepth:
    def test_uint8(self):
        assert _DTYPE_TO_BIT_DEPTH["uint8"] == 8

    def test_uint16(self):
        assert _DTYPE_TO_BIT_DEPTH["uint16"] == 16

    def test_int16(self):
        assert _DTYPE_TO_BIT_DEPTH["int16"] == 16

    def test_float32(self):
        assert _DTYPE_TO_BIT_DEPTH["float32"] == 32


# =====================================================================
# _IntensityAdapter Tests
# =====================================================================
class TestIntensityAdapter:
    """Tests for the _IntensityAdapter wrapper."""

    def test_stamps_bit_depth_uint8(self):
        adapter = _IntensityAdapter(_IdentityTransform(), storage_dtype="uint8")
        img_info = _SimpleImgInfo()
        sample = _SimpleSample(image=torch.rand(3, 8, 8), img_info=img_info)
        result = adapter(sample)  # type: ignore[arg-type]
        assert result.img_info.bit_depth == 8

    def test_stamps_bit_depth_uint16(self):
        adapter = _IntensityAdapter(_IdentityTransform(), storage_dtype="uint16")
        img_info = _SimpleImgInfo()
        sample = _SimpleSample(image=torch.rand(3, 8, 8), img_info=img_info)
        adapter(sample)  # type: ignore[arg-type]
        assert img_info.bit_depth == 16

    def test_stamps_bit_depth_float32(self):
        adapter = _IntensityAdapter(_IdentityTransform(), storage_dtype="float32")
        img_info = _SimpleImgInfo()
        sample = _SimpleSample(image=torch.rand(3, 8, 8), img_info=img_info)
        adapter(sample)  # type: ignore[arg-type]
        assert img_info.bit_depth == 32

    def test_unknown_dtype_defaults_to_8(self):
        adapter = _IntensityAdapter(_IdentityTransform(), storage_dtype="bfloat16")
        assert adapter.bit_depth == 8

    def test_applies_inner_transform(self):
        adapter = _IntensityAdapter(_ScaleTransform(), storage_dtype="uint8")
        img = torch.ones(3, 4, 4)
        sample = _SimpleSample(image=img, img_info=_SimpleImgInfo())
        result = adapter(sample)  # type: ignore[arg-type]
        assert torch.allclose(result.image, torch.full_like(img, 0.5))

    def test_no_img_info_does_not_crash(self):
        """When img_info is None, bit_depth stamping is skipped (no error)."""
        adapter = _IntensityAdapter(_IdentityTransform(), storage_dtype="uint16")
        sample = _SimpleSample(image=torch.rand(3, 4, 4), img_info=None)
        result = adapter(sample)  # type: ignore[arg-type]
        assert result.img_info is None  # no crash

    def test_is_nn_module(self):
        adapter = _IntensityAdapter(_IdentityTransform())
        assert isinstance(adapter, nn.Module)

    def test_inner_transform_accessible(self):
        inner = _IdentityTransform()
        adapter = _IntensityAdapter(inner, storage_dtype="uint8")
        assert adapter.transform is inner


# =====================================================================
# _sanitize_annotations Tests
# =====================================================================
class TestSanitizeAnnotations:
    """Tests for GPUAugmentationPipeline._sanitize_annotations."""

    @pytest.fixture
    def pipeline(self) -> GPUAugmentationPipeline:
        return GPUAugmentationPipeline()

    def test_bboxes_none_returns_all_none(self, pipeline: GPUAugmentationPipeline):
        """When bboxes=None, everything returned unchanged."""
        b, lab, m, k = pipeline._sanitize_annotations(
            torch.rand(2, 3, 32, 32),
            bboxes=None,
            labels=None,
            masks=None,
            keypoints=None,
        )
        assert b is None
        assert lab is None

    def test_clips_bboxes_to_image_bounds(self, pipeline: GPUAugmentationPipeline):
        """Bboxes extending outside image should be clamped."""
        images = torch.rand(1, 3, 64, 64)
        bboxes = [torch.tensor([[-10.0, -5.0, 50.0, 50.0], [20.0, 20.0, 80.0, 80.0]])]
        labels = [torch.tensor([0, 1])]
        out_b, out_l, _, _ = pipeline._sanitize_annotations(images, bboxes, labels, None, None)
        assert out_b is not None
        # First bbox should be clamped to [0, 0, 50, 50]
        # All bboxes coordinates should be within bounds
        for b in out_b:
            if b.numel() > 0:
                assert (b[:, 0] >= 0).all()
                assert (b[:, 1] >= 0).all()
                assert (b[:, 2] <= 64).all()
                assert (b[:, 3] <= 64).all()

    def test_removes_tiny_bboxes(self, pipeline: GPUAugmentationPipeline):
        """Bboxes with width/height below min_size should be filtered out."""
        images = torch.rand(1, 3, 64, 64)
        bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0], [10.0, 10.0, 12.0, 12.0]])]
        labels = [torch.tensor([0, 1])]
        out_b, out_l, _, _ = pipeline._sanitize_annotations(
            images, bboxes, labels, None, None, min_size=4.0, min_area=16.0
        )
        # Second bbox has width=2, height=2 → filtered out
        assert out_b is not None
        assert len(out_b[0]) == 1
        assert out_l is not None
        assert len(out_l[0]) == 1

    def test_labels_filtered_in_lockstep(self, pipeline: GPUAugmentationPipeline):
        """When bboxes are removed, corresponding labels should also be removed."""
        images = torch.rand(1, 3, 64, 64)
        bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0], [10.0, 10.0, 11.0, 11.0]])]
        labels = [torch.tensor([42, 99])]
        out_b, out_l, _, _ = pipeline._sanitize_annotations(images, bboxes, labels, None, None)
        assert out_l is not None
        assert out_l[0].tolist() == [42]

    def test_masks_filtered_with_bboxes(self, pipeline: GPUAugmentationPipeline):
        """Instance masks should be filtered by the same valid mask."""
        images = torch.rand(1, 3, 64, 64)
        bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0], [10.0, 10.0, 11.0, 11.0]])]
        labels = [torch.tensor([0, 1])]
        masks = [torch.rand(2, 64, 64)]  # 2 instance masks
        out_b, out_l, out_m, _ = pipeline._sanitize_annotations(images, bboxes, labels, masks, None)
        assert out_m is not None
        assert out_m[0].shape[0] == 1  # only first mask kept

    def test_empty_bboxes(self, pipeline: GPUAugmentationPipeline):
        """Empty bboxes should not crash."""
        images = torch.rand(1, 3, 32, 32)
        bboxes = [torch.zeros(0, 4)]
        labels = [torch.zeros(0, dtype=torch.long)]
        out_b, out_l, _, _ = pipeline._sanitize_annotations(images, bboxes, labels, None, None)
        assert out_b is not None
        assert len(out_b[0]) == 0

    def test_batch_mismatch_raises(self, pipeline: GPUAugmentationPipeline):
        images = torch.rand(2, 3, 32, 32)
        bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]  # only 1 element
        with pytest.raises(RuntimeError, match="bboxes batch mismatch"):
            pipeline._sanitize_annotations(images, bboxes, None, None, None)

    def test_labels_mismatch_raises(self, pipeline: GPUAugmentationPipeline):
        images = torch.rand(2, 3, 32, 32)
        bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        labels = [torch.tensor([0])]  # only 1 element
        with pytest.raises(RuntimeError, match="labels batch mismatch"):
            pipeline._sanitize_annotations(images, bboxes, labels, None, None)

    def test_bad_bbox_shape_raises(self, pipeline: GPUAugmentationPipeline):
        images = torch.rand(1, 3, 32, 32)
        bboxes = [torch.tensor([10.0, 10.0, 50.0, 50.0])]  # 1D, not 2D
        with pytest.raises(RuntimeError, match="must be .N,4."):
            pipeline._sanitize_annotations(images, bboxes, None, None, None)

    def test_keypoints_clamped(self, pipeline: GPUAugmentationPipeline):
        """Keypoints should be clamped to image bounds."""
        images = torch.rand(1, 3, 64, 64)
        bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        keypoints = [torch.tensor([[-5.0, 70.0]])]  # out of bounds
        _, _, _, out_k = pipeline._sanitize_annotations(images, bboxes, None, None, keypoints)
        assert out_k is not None
        assert (out_k[0][..., 0] >= 0).all()
        assert (out_k[0][..., 1] <= 64).all()

    def test_nonfinite_bboxes_filtered(self, pipeline: GPUAugmentationPipeline):
        """Non-finite (NaN/Inf) bboxes should be removed."""
        images = torch.rand(1, 3, 64, 64)
        bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0], [float("nan"), 10.0, 50.0, 50.0]])]
        labels = [torch.tensor([0, 1])]
        out_b, out_l, _, _ = pipeline._sanitize_annotations(images, bboxes, labels, None, None)
        assert out_b is not None
        assert len(out_b[0]) == 1

    def test_labels_size_mismatch_raises(self, pipeline: GPUAugmentationPipeline):
        """Labels count not matching bboxes count should raise."""
        images = torch.rand(1, 3, 64, 64)
        bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        labels = [torch.tensor([0, 1, 2])]  # 3 labels vs 1 bbox
        with pytest.raises(RuntimeError, match="labels.*size mismatch"):
            pipeline._sanitize_annotations(images, bboxes, labels, None, None)

    def test_semantic_mask_passthrough(self, pipeline: GPUAugmentationPipeline):
        """Semantic masks (shape doesn't match N objects) are passed through unchanged."""
        images = torch.rand(1, 3, 64, 64)
        bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        labels = [torch.tensor([0])]
        # Semantic mask: 3D but first dim != n_bboxes
        masks = [torch.rand(5, 64, 64)]  # 5 != 1
        out_b, _, out_m, _ = pipeline._sanitize_annotations(images, bboxes, labels, masks, None)
        assert out_m is not None
        # Semantic mask passed through unfiltered
        assert out_m[0].shape[0] == 5


# =====================================================================
# GPU Pipeline - Kornia single-key normalisation fix
# =====================================================================
class TestGPUPipelineSingleKeyNormalise:
    """Verify single-data-key Kornia results are normalised to a list."""

    def test_single_key_preserves_batch_dim(self):
        """Regression: single data_key should not lose the batch dimension."""
        import kornia.augmentation as kaug

        pipeline = GPUAugmentationPipeline(
            [kaug.RandomHorizontalFlip(p=0.0)],  # p=0 → identity
            data_keys=["input"],
        )
        images = torch.rand(3, 3, 16, 16)
        result = pipeline(images)
        assert result["images"].shape == (3, 3, 16, 16)

    def test_multi_key_preserves_batch_dim(self):
        """Multi data_key should also preserve batch dimension."""
        import kornia.augmentation as kaug

        pipeline = GPUAugmentationPipeline(
            [kaug.RandomHorizontalFlip(p=0.0)],
            data_keys=["input", "mask"],
        )
        images = torch.rand(2, 3, 16, 16)
        masks = [torch.randint(0, 2, (1, 16, 16), dtype=torch.float32) for _ in range(2)]
        result = pipeline(images, masks=masks)
        assert result["images"].shape == (2, 3, 16, 16)
