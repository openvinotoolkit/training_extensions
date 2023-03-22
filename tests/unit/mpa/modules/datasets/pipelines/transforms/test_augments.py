"""Unit Tests for the MPA Dataset Pipelines Transforms Augments."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from otx.algorithms.common.adapters.mmcv.pipelines.transforms.augments import (
    Augments,
    CythonAugments,
)


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (100, 100), color="red")


class TestAugment:
    @pytest.mark.parametrize(
        "augmentation_str, args",
        [
            ("autocontrast", {}),
            ("equalize", {}),
            ("solarize", {"threshold": 128}),
            ("posterize", {"bits_to_keep": 4}),
            ("posterize", {"bits_to_keep": 8}),
            ("color", {"factor": 0.5}),
            ("contrast", {"factor": 0.5}),
            ("brightness", {"factor": 0.5}),
            ("sharpness", {"factor": 0.5}),
            ("rotate", {"degree": 45}),
            ("shear_x", {"factor": 0.5}),
            ("shear_y", {"factor": 0.5}),
            ("translate_x_rel", {"pct": 0.5}),
            ("translate_y_rel", {"pct": 0.5}),
        ],
    )
    def test_augmentation_function(self, image: Image.Image, augmentation_str: str, args: dict[str, Any]) -> None:
        """Test that the augmentation functions returns an Image object."""
        augmentation_func = getattr(Augments, augmentation_str)
        result = augmentation_func(image, **args)
        assert isinstance(result, Image.Image)

    def test_rotate_with_list_interpolation_instance(self, image: Image.Image) -> None:
        """Test whether list of interpolation instances are accepted."""
        result = Augments.rotate(image, 45, resample=[Image.BICUBIC, Image.BILINEAR])
        assert isinstance(result, Image.Image)


class TestCythonAugments:
    @pytest.mark.parametrize(
        "augmentation_str, args",
        [
            ("autocontrast", {}),
            ("equalize", {}),
            ("solarize", {"threshold": 128}),
            ("posterize", {"bits_to_keep": 4}),
            ("posterize", {"bits_to_keep": 8}),
            ("color", {"factor": 0.5}),
            ("contrast", {"factor": 0.5}),
            ("brightness", {"factor": 0.5}),
            ("sharpness", {"factor": 0.5}),
        ],
    )
    def test_augmentation_output_equals_to_input_image(
        self, image: Image.Image, augmentation_str: str, args: dict[str, Any]
    ) -> None:
        """Test that the augmentation functions returns an Image object."""
        augmentation_func = getattr(CythonAugments, augmentation_str)
        result = augmentation_func(image, **args)
        assert isinstance(result, Image.Image)
        assert result == image

    @pytest.mark.parametrize(
        "augmentation_str, args",
        [
            ("rotate", {"degree": 45}),
            ("shear_x", {"factor": 0.5}),
            ("shear_y", {"factor": 0.5}),
            ("translate_x_rel", {"pct": 0.5}),
            ("translate_y_rel", {"pct": 0.5}),
        ],
    )
    def test_augmentation_output_not_equals_to_input_image(
        self, image: Image.Image, augmentation_str: str, args: dict[str, Any]
    ) -> None:
        """Test that the augmentation functions returns an Image object."""
        augmentation_func = getattr(CythonAugments, augmentation_str)
        result = augmentation_func(image, **args)
        assert isinstance(result, Image.Image)
        assert result != image

    def test_blend(self, image: Image.Image) -> None:
        """Test that it raises an assertion error if dst is not a numpy array."""
        with pytest.raises(AssertionError):
            CythonAugments.blend(image, image, 0.5)
