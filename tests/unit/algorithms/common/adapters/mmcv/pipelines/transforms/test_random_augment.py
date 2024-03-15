"""Unit Tests for the OTX Dataset Pipelines - Random Augment."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from PIL import Image

from otx.algorithms.classification.adapters.mmcls.datasets.pipelines.transforms.random_augment import (
    OTXRandAugment,
    cutout_abs,
    rand_augment_pool,
)


@pytest.fixture
def sample_np_image() -> np.ndarray:
    return np.ones((256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image() -> Image:
    return Image.new("RGB", (256, 256), (255, 255, 255))


def test_all_transforms_return_valid_image(sample_pil_image: Image.Image) -> None:
    """Test all transforms return valid image."""
    for transform, value, max_value in rand_augment_pool:
        img, *extra = transform(sample_pil_image, value=value, max_value=max_value)
        assert isinstance(img, Image.Image)
        assert img.size == sample_pil_image.size


def test_cutoutabs_transform(sample_pil_image: Image.Image) -> None:
    """Test cutout_abs transform."""
    img, (x0, y0, x1, y1), color = cutout_abs(sample_pil_image, 2)
    assert isinstance(img, Image.Image)
    assert img.size == sample_pil_image.size
    assert x0 >= 0 and y0 >= 0
    assert x1 <= sample_pil_image.width and y1 <= sample_pil_image.height
    assert color == (127, 127, 127)


class TestOTXRandAugment:
    def test_with_default_arguments(self, mocker, sample_np_image: np.ndarray) -> None:
        """Test case with default arguments."""
        mocker.patch("random.random", return_value=0.1)  # RandAugment is applied only when random.random() < 0.5
        transform = OTXRandAugment(num_aug=2, magnitude=5, cutout_value=16)
        data = {"img": sample_np_image}
        results = transform(data)

        assert isinstance(results["img"], np.ndarray)
        assert any(item.startswith("rand_mc_") for item in results.keys())
        assert "CutoutAbs" in results

    def test_with_img_fields_argument(self, mocker, sample_np_image: np.ndarray) -> None:
        """Test case with img_fields argument."""
        mocker.patch("random.random", return_value=0.1)  # RandAugment is applied only when random.random() < 0.5
        transform = OTXRandAugment(num_aug=2, magnitude=5, cutout_value=16)
        data = {
            "img1": sample_np_image,
            "img2": sample_np_image,
            "img_fields": ["img1"],
        }
        results = transform(data)
        assert isinstance(results["img1"], np.ndarray)
        assert any(item.startswith("rand_mc_") for item in results.keys())
        assert "CutoutAbs" in results

    def test_with_pil_image_input(self, sample_pil_image: Image.Image) -> None:
        """Test case with PIL.Image input."""
        transform = OTXRandAugment(num_aug=2, magnitude=5, cutout_value=16)
        data = {"img": sample_pil_image}
        results = transform(data)

        assert isinstance(results["img"], np.ndarray)
        assert "CutoutAbs" in results
