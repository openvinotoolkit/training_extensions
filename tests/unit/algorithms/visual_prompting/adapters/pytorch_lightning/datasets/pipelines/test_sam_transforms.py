"""Tests sam transforms used for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines.sam_transforms import (
    ResizeLongestSide,
)

from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestResizeLongestSide:
    @e2e_pytest_unit
    def test_apply_boxes(self):
        """Test apply_boxes."""
        resize_longest_side = ResizeLongestSide(100)
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        original_size = (200, 200)
        expected_result = np.array([[5, 10, 15, 20], [25, 30, 35, 40]])

        result = resize_longest_side.apply_boxes(boxes, original_size)

        assert np.array_equal(result, expected_result)

    @e2e_pytest_unit
    def test_apply_image_torch(self):
        """Test apply_image_torch."""
        resize_longest_side = ResizeLongestSide(100)
        image = torch.zeros((1, 3, 200, 300), dtype=torch.float32)
        expected_result_shape = (1, 3, 67, 100)

        result = resize_longest_side.apply_image_torch(image)

        assert result.shape == expected_result_shape

    @e2e_pytest_unit
    def test_apply_coords_torch(self):
        """Test apply_coords_torch."""
        resize_longest_side = ResizeLongestSide(100)
        coords = torch.Tensor([[50, 50], [100, 100]])
        original_size = (200, 200)
        expected_result = torch.Tensor([[25, 25], [50, 50]])

        result = resize_longest_side.apply_coords_torch(coords, original_size)

        assert torch.allclose(result, expected_result)

    @e2e_pytest_unit
    def test_apply_boxes_torch(self):
        """Test apply_boxes_torch."""
        resize_longest_side = ResizeLongestSide(100)
        boxes = torch.Tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        original_size = (200, 200)
        expected_result = torch.Tensor([[5, 10, 15, 20], [25, 30, 35, 40]])

        result = resize_longest_side.apply_boxes_torch(boxes, original_size)

        assert torch.allclose(result, expected_result)

    @e2e_pytest_unit
    def test_get_preprocess_shape(self):
        """Test get_preprocess_shape."""
        resize_longest_side = ResizeLongestSide(100)
        oldh, oldw = 200, 300
        expected_result = (67, 100)

        result = resize_longest_side.get_preprocess_shape(oldh, oldw, resize_longest_side.target_length)

        assert result == expected_result
