# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for torchvision kernel registrations on ImageInfo."""

from __future__ import annotations

import pytest

from getitune.data.augmentation.kernels import (
    _center_crop_image_info,
    _crop_image_info,
    _pad_image_info,
    _resize_image_info,
    _resized_crop_image_info,
)
from getitune.data.entity.base import ImageInfo


def _make_img_info(h: int = 100, w: int = 200) -> ImageInfo:
    """Create an ImageInfo with the given original dimensions."""
    return ImageInfo(img_idx=0, img_shape=(h, w), ori_shape=(h, w))  # type: ignore[no-matching-overload]


class TestResizeImageInfo:
    """Tests for _resize_image_info kernel."""

    def test_resize_two_element_size(self):
        info = _make_img_info(100, 200)
        result = _resize_image_info(info, [50, 100])
        assert result.img_shape == (50, 100)
        assert result.scale_factor == pytest.approx((0.5, 0.5))

    def test_resize_single_element_size(self):
        info = _make_img_info(100, 200)
        result = _resize_image_info(info, [64])
        assert result.img_shape == (64, 64)

    def test_resize_invalid_size_raises(self):
        info = _make_img_info()
        with pytest.raises(ValueError, match=r"\[1, 2, 3\]"):
            _resize_image_info(info, [1, 2, 3])

    def test_resize_scale_factor(self):
        info = _make_img_info(100, 200)
        _resize_image_info(info, [200, 400])
        assert info.scale_factor == pytest.approx((2.0, 2.0))


class TestCropImageInfo:
    """Tests for _crop_image_info kernel."""

    def test_crop_updates_shape(self):
        info = _make_img_info(100, 200)
        result = _crop_image_info(info, height=50, width=80)
        assert result.img_shape == (50, 80)

    def test_crop_clears_scale_factor(self):
        info = _make_img_info(100, 200)
        _crop_image_info(info, height=50, width=80)
        assert info.scale_factor is None


class TestResizedCropImageInfo:
    """Tests for _resized_crop_image_info kernel."""

    def test_resized_crop_two_element(self):
        info = _make_img_info()
        result = _resized_crop_image_info(info, [128, 256])
        assert result.img_shape == (128, 256)
        assert result.scale_factor is None

    def test_resized_crop_one_element(self):
        info = _make_img_info()
        result = _resized_crop_image_info(info, [64])
        assert result.img_shape == (64, 64)

    def test_resized_crop_invalid_raises(self):
        info = _make_img_info()
        with pytest.raises(ValueError, match=r"\[1, 2, 3\]"):
            _resized_crop_image_info(info, [1, 2, 3])


class TestCenterCropImageInfo:
    """Tests for _center_crop_image_info kernel."""

    def test_center_crop(self):
        info = _make_img_info(100, 200)
        result = _center_crop_image_info(info, output_size=[50, 80])
        assert result.img_shape == (50, 80)
        assert result.scale_factor is None


class TestPadImageInfo:
    """Tests for _pad_image_info kernel."""

    def test_pad_int(self):
        info = _make_img_info(100, 200)
        result = _pad_image_info(info, padding=10)
        # int padding → all sides = 10
        assert result.img_shape == (120, 220)

    def test_pad_list_two(self):
        info = _make_img_info(100, 200)
        result = _pad_image_info(info, padding=[5, 10])
        assert result.img_shape == (120, 210)

    def test_pad_list_four(self):
        info = _make_img_info(100, 200)
        result = _pad_image_info(info, padding=[1, 2, 3, 4])
        assert result.img_shape == (106, 204)
        assert result.padding == (1, 2, 3, 4)
