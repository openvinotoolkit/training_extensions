"""Unit Tests for the MPA Dataset Pipeline Torchvision to MMDet."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter
from torch import Tensor

from otx.algorithms.detection.adapters.mmdet.datasets.pipelines.torchvision2mmdet import (
    BranchImage,
    ColorJitter,
    NDArrayToPILImage,
    NDArrayToTensor,
    PILImageToNDArray,
    RandomApply,
    RandomErasing,
    RandomGaussianBlur,
    RandomGrayscale,
)


@pytest.fixture
def data() -> dict[str, np.ndarray]:
    return {"img": np.ones((256, 256, 3), dtype=np.uint8)}


@pytest.fixture()
def image_tensor() -> Tensor:
    return torch.rand(3, 256, 256)


class TestColorJitter:
    def test_call(self, data: dict[str, np.ndarray]) -> None:
        """Test __call__ method of ColorJitter."""
        transform = ColorJitter()
        outputs = transform(data)
        assert outputs.keys() == data.keys()
        assert np.array_equal(outputs["img"], data["img"])

    def test_repr(self) -> None:
        """Test __repr__ method of ColorJitter."""
        transform = ColorJitter(brightness=0.2)
        assert str(transform) == "ColorJitter(brightness=[0.8, 1.2], contrast=None, saturation=None, hue=None)"


class TestRandomGrayscale:
    def test_random_grayscale(self, image_tensor: Tensor) -> None:
        """Test random grayscale."""
        inputs = {"img": image_tensor}
        pipeline = RandomGrayscale(p=0.5)
        outputs = pipeline.forward(inputs)

        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == set(inputs.keys())
        assert outputs["img"].shape == inputs["img"].shape


class TestRandomErasing:
    def test_random_erasing(self, image_tensor: Tensor) -> None:
        """Test random erasing."""
        transform = RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

        data = {"img": image_tensor}
        transformed_data = transform(data)

        assert "img" in transformed_data
        assert transformed_data["img"].shape == data["img"].shape


class TestRandomGaussianBlur:
    def test_random_gaussian_blur(self) -> None:
        """Test initialization."""
        sigma_min = 0.1
        sigma_max = 2.0
        pipeline = RandomGaussianBlur(sigma_min, sigma_max)
        assert pipeline.sigma_min == sigma_min
        assert pipeline.sigma_max == sigma_max

        # Test forward pass
        inputs = {"img": Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))}
        outputs = pipeline(inputs)
        assert outputs.keys() == inputs.keys()
        assert isinstance(outputs["img"], Image.Image)

        # Test that the output image is blurred
        blur_radius = (pipeline.sigma_min + pipeline.sigma_max) / 2
        blurred_image = inputs["img"].filter(ImageFilter.GaussianBlur(radius=blur_radius))
        assert np.array_equal(np.array(outputs["img"]), np.array(blurred_image))

    def test_repr(self) -> None:
        """Test __repr__ method of RandomGaussianBlur."""
        pipeline = RandomGaussianBlur(0.1, 2.0)
        assert repr(pipeline) == "RandomGaussianBlur"


class TestRandomApply:
    def test_random_apply_with(self) -> None:
        """Test RandomApply with a single transform."""
        # Define the transforms to be applied randomly
        transform_cfgs = [
            dict(
                type="ColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            )
        ]

        # Create the RandomApply pipeline
        random_apply = RandomApply(transform_cfgs, p=0.0)

        # Define the inputs and expected outputs
        inputs = {"img": Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8))}
        results = random_apply(inputs)
        assert np.allclose(np.array(results["img"]), np.array(inputs["img"]))


class TestNDArrayToTensor:
    def test_ndarray_to_tensor_with_single_channel_image(self, data: dict[str, np.ndarray]) -> None:
        """Test NDArrayToTensor with a single channel image."""
        pipeline = NDArrayToTensor(keys=["img"])
        output = pipeline(data)

        assert output["img"].shape == (3, 256, 256)
        assert isinstance(output["img"], torch.Tensor)


class TestNDArrayToPILImage:
    def test_ndarray_to_pil_conversion(self, data: dict[str, np.ndarray]) -> None:
        """Test NDArrayToPILImage with a three channel image."""
        pipeline = NDArrayToPILImage(keys=["img"])
        output = pipeline(data)

        assert isinstance(output["img"], Image.Image)
        assert output["img"].size == (256, 256)

    def test_rept(self) -> None:
        """Test __repr__ method of NDArrayToPILImage."""
        pipeline = NDArrayToPILImage(keys=["image"])
        assert repr(pipeline) == "NDArrayToPILImage"


class TestPILImageToNDArray:
    def test_call(self, data: dict[str, np.ndarray]) -> None:
        """Test __call__ method of PILImageToNDArray."""
        pipeline = PILImageToNDArray(keys=["image"])
        data = {"image": Image.fromarray(data["img"])}
        output = pipeline(data)

        assert isinstance(output["image"], np.ndarray)
        assert output["image"].shape == (256, 256, 3)

    def test_repr(self) -> None:
        """Test __repr__ method of PILImageToNDArray."""
        pipeline = PILImageToNDArray(keys=["image"])
        assert repr(pipeline) == "PILImageToNDArray"


class TestBranchImage:
    def test_branch_image(self) -> None:
        """Test BranchImage pipeline."""
        # Test data
        data = {"img": "test.jpg", "label": 0, "img_fields": ["img"]}

        # Call the pipeline
        key_map = {"img": "img2", "label": "label2"}
        pipeline = BranchImage(key_map)
        results = pipeline(data)

        # Check that the results have been updated correctly
        assert results["img2"] == "test.jpg"
        assert results["label2"] == 0
        assert "img2" in data["img_fields"]
        assert "label2" not in data["img_fields"]

    def test_repr(self) -> None:
        """Test __repr__ method of BranchImage."""
        pipeline = BranchImage()
        assert repr(pipeline) == "BranchImage"
