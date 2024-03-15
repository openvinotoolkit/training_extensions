"""Unit Tests for the OTX Dataset Pipeline - Compose."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import RandomCrop

from otx.algorithms.segmentation.adapters.mmseg.datasets import MaskCompose, ProbCompose


class TestProbCompose:
    """ProbCompose Unit Tests."""

    def test_inputs(self) -> None:
        """Test when all inputs are correct."""
        transforms = [MagicMock(), MagicMock(), MagicMock()]
        probs = [0.3, 0.4, 0.3]
        prob_compose = ProbCompose(transforms, probs)
        assert prob_compose.transforms == transforms
        assert np.array_equal(prob_compose.limits, np.array([0.0, 0.3, 0.7, 1.0]))

        # Test when transforms and probs have different lengths
        transforms = [MagicMock(), MagicMock()]
        probs = [0.5, 0.5, 0.5]
        with pytest.raises(AssertionError):
            ProbCompose(transforms, probs)

        # Test when probs have negative values
        transforms = [MagicMock(), MagicMock(), MagicMock()]
        probs = [0.5, -0.2, 0.7]
        with pytest.raises(AssertionError):
            ProbCompose(transforms, probs)

        # Test when sum of probs is 0
        transforms = [MagicMock(), MagicMock(), MagicMock()]
        probs = [0.0, 0.0, 0.0]
        with pytest.raises(AssertionError):
            ProbCompose(transforms, probs)

    def test_dict_transforms(self) -> None:
        """Test whether dict transforms are correctly appended."""
        prob_compose = ProbCompose(transforms=[dict(type="Resize")], probs=[0.7])
        assert (
            repr(prob_compose.transforms[0])
            == "Resize(img_scale=None, multiscale_mode=range, ratio_range=None, keep_ratio=True)"
        )

    def test_invalid_transform_type(self) -> None:
        """Test invalid transform type raises error."""
        with pytest.raises(TypeError):
            transforms = ["Dummy Transform"]
            probs = [0.5]
            pipeline = ProbCompose(transforms, probs)
            del pipeline  # This is to silence the pylint's unused variable warning.

    def test_repr(self) -> None:
        """Test the repr method."""
        transforms = [MagicMock()]
        probs = [0.3]
        prob_compose = ProbCompose(transforms, probs)
        expected_repr = f"ProbCompose(\n    {transforms[0]}\n)"
        assert repr(prob_compose) == expected_repr


@PIPELINES.register_module()
class TestTransform(object):
    def __init__(self) -> None:
        pass

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return data


@pytest.fixture()
def data() -> dict[str, np.ndarray]:
    return {"img": np.ones((100, 100, 3), dtype=np.uint8), "label": np.ones((100, 100))}


class TestMaskCompose:
    """MaskCompose Unit Tests."""

    def test_keep_original_false(self, data: dict[str, np.ndarray]) -> None:
        """Test that the main image is replaced with a mixed image when keep_original is False."""
        transforms = [dict(type="TestTransform")]
        pipeline = MaskCompose(transforms=transforms, prob=1.0, keep_original=False)
        mixed_data = pipeline(data)

        assert np.array_equal(data["img"], mixed_data["img"])
        assert np.array_equal(data["label"], mixed_data["label"])

    def test_keep_original_true(self, data: dict[str, np.ndarray]) -> None:
        """Test that the mixed image is added as aux_img when keep_original is True."""
        transforms = [dict(type="TestTransform")]
        pipeline = MaskCompose(transforms=transforms, prob=1.0, keep_original=True)
        mixed_data = pipeline(data)

        assert np.array_equal(mixed_data["img"], mixed_data["aux_img"])
        assert np.array_equal(data["img"], mixed_data["img"])
        assert np.array_equal(data["label"], mixed_data["label"])

    def test_callable_transform(self, data: dict[str, np.ndarray]) -> None:
        """Test callable transform is appended to the list of transforms."""
        crop_size = (10, 10)
        transform = RandomCrop(crop_size=crop_size)
        pipeline = MaskCompose(transforms=[transform], prob=1.0, keep_original=False)
        mixed_data = pipeline(data)
        assert mixed_data["img_shape"][:2] == crop_size

    def test_invalid_transform_type(self) -> None:
        """Test invalid transform type raises error."""
        with pytest.raises(TypeError):
            transform = [{"Dummy Transform": "So dummy"}]
            pipeline = MaskCompose(transforms=[transform], prob=1.0, keep_original=False)
            del pipeline  # This is to silence the pylint's unused variable warning.

    def test_prob_zero(self, data: dict[str, np.ndarray]) -> None:
        """Test that the main image is not modified when prob is 0."""
        transforms = [
            dict(type="RandomFlip", prob=0.0, direction="horizontal"),
            dict(type="RandomRotate", prob=0.0, degree=30, pad_val=0, seg_pad_val=255),
        ]
        pipeline = MaskCompose(transforms=transforms, prob=0, keep_original=False)
        mixed_data = pipeline(data)

        assert np.array_equal(data["img"], mixed_data["img"])
        assert np.array_equal(data["label"], mixed_data["label"])

    def test_apply_transforms_returns_none(self, data: dict[str, np.ndarray]) -> None:
        """Test that None is returned when apply_transforms returns None."""
        transforms = [
            dict(type="RandomFlip", prob=0.5, direction="horizontal"),
            lambda x: None,
        ]
        pipeline = MaskCompose(transforms=transforms, prob=1.0, keep_original=False)

        with pytest.raises(AssertionError):
            mixed_data = pipeline(data)
            del mixed_data  # This is to silence the pylint's unused variable warning.

    def test_repr(self) -> None:
        """Test __repr__ method."""
        pipeline = MaskCompose(transforms=[dict(type="RandomFlip")], prob=1.0)
        expected_repr = "MaskCompose(\n    RandomFlip(prob=None)\n)"
        assert repr(pipeline) == expected_repr
