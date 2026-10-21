# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for polygon-native vs. mask-based CPU transforms.

Every test runs the same geometry twice through identically seeded transforms:
1. Polygon path: keeps exact float coordinates, rasterizing once at final resolution.
2. Mask path: rasterizes up front at source resolution, nearest-resampling the bitmap
   through each geometric stage.

Why Hausdorff Distance Instead of IoU
------------------------------------
Because the two paths quantise at different points, they disagree by ~1px along boundaries.
A fixed IoU floor is unreliable because IoU scales with perimeter-to-area ratio (e.g., failing
on small triangles while passing on large squares) and shifts with random crops.

Instead, :func:`assert_band_equal` enforces a bounded **Hausdorff distance** (each mask
must lie inside the other dilated by `k` pixels). This is scale-free and fails loudly on
systemic bugs like wrong pad offsets or dropped ring separators.

Choosing k
----------
`k` tracks the number of resampling stages, as each contributes up to ~1px  of displacement:
- **k=1**: Exact transforms (flips) and a single resize.
- **k=2**: One cached transform (mosaic letterbox + perspective crop).
- **k=3**: Full pipeline (resize, mosaic, and crop resize).

Raising `k` beyond the stage count hides regressions; lowering it makes the test flaky.
"""

from __future__ import annotations

import pytest
import torch
import torchvision.transforms.v2 as tvt_v2
from torchvision import tv_tensors

from getitune.data.augmentation.cache import _clone_for_cache
from getitune.data.augmentation.pipeline import CPUAugmentationPipeline
from getitune.data.augmentation.transforms import (
    CachedMixUp,
    CachedMosaic,
    FilterBoundingBoxes,
    Resize,
)
from getitune.data.entity import BaseSample
from getitune.data.entity.rasterization import rasterize


def _dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    """Dilates a binary mask by `k` pixels using max pooling.

    Args:
        m (torch.Tensor): A binary input tensor (N, H, W).
        k (int): Number of dilation iterations, corresponding to the pixel
            radius expansion.

    Returns:
        torch.Tensor: The dilated boolean tensor of the same shape as `m`.
    """
    x = m.unsqueeze(1).float()
    for _ in range(k):
        x = torch.nn.functional.max_pool2d(x, 3, stride=1, padding=1)
    return x.squeeze(1).bool()


def assert_band_equal(a: torch.Tensor, b: torch.Tensor, k: int) -> None:
    """Asserts that two segmentation masks are equivalent within a k-pixel
    Hausdorff distance tolerance band.

    This is particularly useful for verifying data augmentation pipelines where
    rasterized ground-truth polygons and transformed masks may exhibit minor
    sub-pixel rounding or anti-aliasing discrepancies.

    Args:
        a (torch.Tensor): The first mask tensor (e.g., rasterized polygon).
        b (torch.Tensor): The second mask tensor to compare against
            (e.g., augmented mask).
        k (int): The maximum allowed pixel tolerance distance.

    Raises:
        AssertionError: If any foreground pixel in `a` falls outside the
            k-pixel dilation of `b`, or vice versa.
    """
    assert (a.bool() & ~_dilate(b.bool(), k)).flatten(1).sum(1).max() == 0
    assert (b.bool() & ~_dilate(a.bool(), k)).flatten(1).sum(1).max() == 0


class TestResizeCommutativity:
    @pytest.mark.parametrize(
        ("src_hw", "dst_hw", "keep_aspect_ratio", "center_padding"),
        [
            ((100, 100), (200, 200), False, False),  # pure upscale
            ((256, 256), (128, 128), False, False),  # pure downscale
            ((100, 160), (224, 224), True, False),  # letterbox, bottom-right pad
            ((160, 100), (224, 224), True, True),  # letterbox, centred pad
        ],
    )
    def test_resize_commutes(
        self,
        fxt_polygon_sample,
        fxt_mask_sample,
        src_hw,
        dst_hw,
        keep_aspect_ratio,
        center_padding,
    ):
        kwargs = {"size": dst_hw, "keep_aspect_ratio": keep_aspect_ratio, "center_padding": center_padding}
        poly_out = Resize(**kwargs)(fxt_polygon_sample(src_hw))
        mask_out = Resize(**kwargs)(fxt_mask_sample(src_hw))

        rasterized = rasterize(poly_out.polygons, canvas_size=poly_out.image.shape[-2:])
        assert rasterized.shape == mask_out.masks.shape
        assert_band_equal(rasterized, mask_out.masks, k=2)

    def test_early_exit_leaves_polygons_identical(self, fxt_polygon_sample):
        sample = fxt_polygon_sample((100, 100))
        before = sample.polygons.clone()
        after = Resize(size=(100, 100))(sample).polygons
        torch.testing.assert_close(
            after.as_subclass(torch.Tensor),
            before.as_subclass(torch.Tensor),
            equal_nan=True,
        )
        assert tuple(after.canvas_size) == tuple(before.canvas_size)

    def test_zero_instances(self, fxt_polygon_sample):
        out = Resize(size=(64, 64))(fxt_polygon_sample((100, 100), rings=[]))
        assert out.polygons.shape[0] == 0
        assert rasterize(out.polygons, (64, 64)).shape == (0, 64, 64)

    def test_nan_padding_survives(self, fxt_polygon_sample):
        """Ragged instances keep their trailing NaN rows -- otherwise padding
        becomes a real vertex and the triangle grows extra edges."""
        out = Resize(size=(200, 200))(fxt_polygon_sample((100, 100)))
        assert torch.isnan(out.polygons[0, 3:]).all()

    def test_hole_preserved(self, fxt_polygon_sample, fxt_polygon_rings):
        """Ring separators must survive scaling, or the hole fills solid."""
        out = Resize(size=(200, 200))(fxt_polygon_sample((100, 100)))
        mask = rasterize(out.polygons, (200, 200))
        assert mask[2, 140, 44] == 0  # inside the hole
        assert mask[2, 110, 25] == 1  # inside the outer ring


class TestCachedMosaicPolygons:
    def _run(self, transform_factory, sample_factory, seed=7) -> BaseSample:
        """Warm the cache with 4 samples, then run one seeded forward."""
        tf = transform_factory()
        torch.manual_seed(seed)
        for i in range(4):
            tf(sample_factory((100, 100), seed=i))
        torch.manual_seed(seed)
        return tf(sample_factory((100, 100), seed=99))

    def test_mosaic_parity(self, fxt_polygon_sample, fxt_mask_sample):
        factory = lambda: CachedMosaic(img_scale=(128, 128), p=1.0, max_cached_images=4)
        poly_out = self._run(factory, fxt_polygon_sample)
        mask_out = self._run(factory, fxt_mask_sample)
        rasterized = rasterize(poly_out.polygons, canvas_size=(128, 128))
        assert rasterized.shape[0] == mask_out.masks.shape[0] == poly_out.bboxes.shape[0]
        assert_band_equal(rasterized, mask_out.masks, k=2)

    def test_non_mosaic_letterbox_path_parity(self, fxt_polygon_sample, fxt_mask_sample):
        factory = lambda: CachedMosaic(img_scale=(128, 128), p=0.0, max_cached_images=4)
        poly_out = self._run(factory, fxt_polygon_sample)
        mask_out = self._run(factory, fxt_mask_sample)
        rasterized = rasterize(poly_out.polygons, canvas_size=poly_out.image.shape[-2:])
        assert rasterized.shape == mask_out.masks.shape
        assert_band_equal(rasterized, mask_out.masks, k=2)

    def test_empty_sample_survives_mosaic(self, fxt_polygon_sample):
        tf = CachedMosaic(img_scale=(128, 128), p=1.0, max_cached_images=4)
        for i in range(4):
            tf(fxt_polygon_sample((100, 100), rings=[], seed=i))
        out = tf(fxt_polygon_sample((100, 100), rings=[], seed=99))
        assert out.polygons.shape[0] == out.bboxes.shape[0] == 0
        assert tuple(out.polygons.canvas_size) == (128, 128)


class TestCachedMixUpPolygons:
    def _run(self, sample_factory, seed=5) -> BaseSample:
        """Warm the cache with 4 samples, then run one seeded forward."""
        tf = CachedMixUp(img_scale=(128, 128), p=1.0, max_cached_images=4)
        torch.manual_seed(seed)
        for i in range(4):
            tf(sample_factory((100, 100), seed=i))
        torch.manual_seed(seed)
        return tf(sample_factory((100, 100), seed=99))

    def test_mixup_parity(self, fxt_polygon_sample, fxt_mask_sample):
        poly_out = self._run(fxt_polygon_sample)
        mask_out = self._run(fxt_mask_sample)
        rasterized = rasterize(poly_out.polygons, canvas_size=poly_out.image.shape[-2:])
        assert rasterized.shape == mask_out.masks.shape
        assert_band_equal(rasterized, mask_out.masks, k=2)

    def test_concat_pads_ragged_pmax(self, fxt_polygon_sample, fxt_polygon_rings, mocker):
        """Cached sample has 9-vertex rings, current sample only a 3-vertex
        triangle: concat must pad to the larger P_max, not raise."""
        tf = CachedMixUp(img_scale=(128, 128), p=1.0, max_cached_images=4)
        tf(fxt_polygon_sample((100, 100)))  # P_max = 9
        # forward() appends the current sample to the cache before blending, so
        # without pinning the index MixUp may pick the 3-vertex sample itself.
        mocker.patch.object(tf, "_get_cached_index", return_value=0)
        out = tf(fxt_polygon_sample((100, 100), rings=fxt_polygon_rings[:1]))  # P_max = 3

        assert out.polygons.shape[0] == out.bboxes.shape[0] == out.label.shape[0]
        assert out.polygons.shape[1] == 9
        # The 3-vertex instance must be NaN-padded, not zero-filled: a real
        # (0, 0) vertex would drag an extra edge to the origin on rasterize.
        assert torch.isnan(out.polygons[0, 3:]).all()

    def test_cached_sample_rescaled(self, fxt_polygon_sample):
        """Differing canvas sizes: cached polygons are scaled by min(h,w) ratio
        (transforms.py:1016) -- must land inside the current canvas."""
        tf = CachedMixUp(img_scale=(128, 128), p=1.0, max_cached_images=4)
        tf(fxt_polygon_sample((200, 200)))
        out = tf(fxt_polygon_sample((100, 100)))
        finite = out.polygons[torch.isfinite(out.polygons).all(-1)]
        h, w = out.image.shape[-2:]
        assert (finite[:, 0] <= w).all()
        assert (finite[:, 1] <= h).all()

    def test_skips_when_cache_cold(self, fxt_polygon_sample):
        sample = fxt_polygon_sample((100, 100))
        before = sample.polygons.clone()
        out = CachedMixUp(img_scale=(128, 128), p=1.0, max_cached_images=4)(sample)
        torch.testing.assert_close(
            out.polygons.as_subclass(torch.Tensor), before.as_subclass(torch.Tensor), equal_nan=True
        )


class TestFilterBoundingBoxes:
    def test_polygons_filtered_in_lockstep(self, fxt_polygon_sample):
        sample = fxt_polygon_sample((100, 100))
        sample.bboxes[0] = torch.tensor([10.0, 10.0, 10.5, 10.5])  # degenerate
        expected = sample.polygons[1, 0].clone()
        out = FilterBoundingBoxes(min_wh=2)(sample)
        assert out.polygons.shape[0] == out.bboxes.shape[0] == out.label.shape[0] == 2
        torch.testing.assert_close(out.polygons[0, 0].as_subclass(torch.Tensor), expected.as_subclass(torch.Tensor))


class TestCloneForCache:
    def test_polygons_cloned_and_independent(self, fxt_polygon_sample):
        sample = fxt_polygon_sample((100, 100))
        cached = _clone_for_cache(sample)
        sample.polygons[0, 0, 0] = -999.0
        assert cached.polygons[0, 0, 0] != -999.0

    def test_masks_not_materialised_for_polygon_sample(self, fxt_polygon_sample):
        """Cloning masks here would reinstate the full-res copy Phase 2 removes."""
        assert _clone_for_cache(fxt_polygon_sample((100, 100))).masks is None

    def test_oversized_sample_rescales_polygons(self, fxt_polygon_sample, fxt_polygon_rings):
        cached = _clone_for_cache(fxt_polygon_sample((1200, 800)), max_size=1024)
        assert cached.image.shape[-2:] == (1024, 683)
        assert tuple(cached.polygons.canvas_size) == (1024, 683)

        # Instance 1, vertex 0, on the 100x100 reference canvas.
        ref_x, ref_y = fxt_polygon_rings[1][0]
        # The fixture scales the reference to 1200x800 (x *= 8.0, y *= 12.0),
        # then the cache shrinks the longest side to 1024 (x *= 683/800, y *= 1024/1200).
        assert cached.polygons[1, 0, 0] == pytest.approx(ref_x * 8.0 * 683 / 800, rel=1e-3)
        assert cached.polygons[1, 0, 1] == pytest.approx(ref_y * 12.0 * 1024 / 1200, rel=1e-3)


class TestNativeTransformDispatch:
    def test_polygons_passed_to_native_transform(self, fxt_polygon_sample):
        pipeline = CPUAugmentationPipeline([tvt_v2.Resize(size=(50, 50))])
        out = pipeline._apply_native_transform(tvt_v2.Resize(size=(50, 50)), fxt_polygon_sample((100, 100)))
        assert isinstance(out.polygons, tv_tensors.KeyPoints)
        assert out.polygons[1, 0, 0] == pytest.approx(20.0)

    def test_native_flip_parity(self, fxt_polygon_sample, fxt_mask_sample):
        pipeline = CPUAugmentationPipeline([])
        flip = tvt_v2.RandomHorizontalFlip(p=1.0)
        poly_out = pipeline._apply_native_transform(flip, fxt_polygon_sample((100, 100)))
        mask_out = pipeline._apply_native_transform(flip, fxt_mask_sample((100, 100)))
        rasterized = rasterize(poly_out.polygons, canvas_size=(100, 100))
        assert_band_equal(rasterized, mask_out.masks, k=2)


class TestEndToEndPipelineParity:
    @staticmethod
    def _make_pipeline() -> CPUAugmentationPipeline:
        """Resize -> Mosaic -> MixUp -> Filter, identical for both schemas."""
        return CPUAugmentationPipeline(
            [
                Resize(size=(128, 128), keep_aspect_ratio=True),
                CachedMosaic(img_scale=(128, 128), p=1.0, max_cached_images=4),
                CachedMixUp(img_scale=(128, 128), p=1.0, max_cached_images=4),
                FilterBoundingBoxes(min_wh=2),
            ]
        )

    def _run(self, sample_factory, seed=11) -> BaseSample:
        """Warm every cache with the same 4 samples, then one seeded forward."""
        pipeline = self._make_pipeline()
        torch.manual_seed(seed)
        for i in range(4):
            pipeline(sample_factory((100, 100), seed=i))
        torch.manual_seed(seed)
        return pipeline(sample_factory((100, 100), seed=99))

    def test_full_cpu_pipeline_parity(self, fxt_polygon_sample, fxt_mask_sample):
        """Resize + Mosaic + MixUp + Filter, seeded, polygon vs mask."""
        poly_out = self._run(fxt_polygon_sample)
        mask_out = self._run(fxt_mask_sample)

        # Same RNG stream => same geometric decisions => same surviving instances.
        assert poly_out.polygons.shape[0] == mask_out.masks.shape[0]
        assert poly_out.bboxes.shape[0] == mask_out.bboxes.shape[0]
        torch.testing.assert_close(
            poly_out.bboxes.as_subclass(torch.Tensor),
            mask_out.bboxes.as_subclass(torch.Tensor),
            rtol=1e-4,
            atol=1e-3,
        )
        torch.testing.assert_close(poly_out.image, mask_out.image)

        rasterized = rasterize(poly_out.polygons, canvas_size=poly_out.image.shape[-2:])
        assert rasterized.shape == mask_out.masks.shape
        assert_band_equal(rasterized, mask_out.masks, k=3)

    def test_pipeline_output_is_collate_ready(self, fxt_polygon_sample):
        """Every annotation field must stay index-aligned after four stacked
        transforms -- the one invariant collate cannot recover from."""
        out = self._run(fxt_polygon_sample)
        n = out.polygons.shape[0]
        assert out.bboxes.shape[0] == out.label.shape[0] == n
        assert tuple(out.polygons.canvas_size) == tuple(out.image.shape[-2:]) == (128, 128)
        assert getattr(out, "masks", None) is None  # polygon schema never materialises masks
