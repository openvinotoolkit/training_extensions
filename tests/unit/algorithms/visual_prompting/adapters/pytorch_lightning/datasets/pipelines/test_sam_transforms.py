"""Tests sam transforms used for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from typing import Tuple
import pytest
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines.sam_transforms import (
    ResizeLongestSide,
)

from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestResizeLongestSide:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.resize_longest_side = ResizeLongestSide(8)

    @e2e_pytest_unit
    def test_call(self):
        """Test __call__."""

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "image,expected",
        [
            (np.zeros((2, 4, 3), dtype=np.uint8), (4, 8, 3)),
            (np.zeros((12, 16, 3), dtype=np.uint8), (6, 8, 3)),
        ],
    )
    def test_apply_image(self, image: np.ndarray, expected: Tuple[int, int, int]):
        """Test apply_image."""
        results = self.resize_longest_side.apply_image(image, self.resize_longest_side.target_length)

        assert results.shape == expected

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "coords,original_size,expected",
        [
            (np.array([[1, 1], [2, 2]]), (4, 4), np.array([[2, 2], [4, 4]])),
            (np.array([[4, 4], [8, 8]]), (16, 16), np.array([[2, 2], [4, 4]])),
        ],
    )
    def test_apply_coords(self, coords: np.ndarray, original_size: Tuple[int, int], expected: np.ndarray):
        """Test apply_coords."""
        result = self.resize_longest_side.apply_coords(coords, original_size)

        assert np.array_equal(result, expected)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "boxes,original_size,expected",
        [
            (np.array([[1, 1, 2, 2], [2, 2, 3, 3]]), (4, 4), np.array([[2, 2, 4, 4], [4, 4, 6, 6]])),
            (np.array([[4, 4, 8, 8], [8, 8, 12, 12]]), (16, 16), np.array([[2, 2, 4, 4], [4, 4, 6, 6]])),
        ],
    )
    def test_apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, int], expected: np.ndarray):
        """Test apply_boxes."""
        result = self.resize_longest_side.apply_boxes(boxes, original_size)

        assert np.array_equal(result, expected)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "oldh,oldw,expected",
        [
            (3, 4, (6, 8)),
            (12, 16, (6, 8)),
        ],
    )
    def test_get_preprocess_shape(self, oldh: int, oldw: int, expected: Tuple[int, int]):
        """Test get_preprocess_shape."""
        result = self.resize_longest_side.get_preprocess_shape(oldh, oldw, self.resize_longest_side.target_length)

        assert result == expected
