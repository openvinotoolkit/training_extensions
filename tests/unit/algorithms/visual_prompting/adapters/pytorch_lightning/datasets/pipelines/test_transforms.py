"""Tests transforms used for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, Tuple
from unittest import mock

import numpy as np
import pytest
import torch
from torch import Tensor
from torchvision.transforms import Normalize

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines.sam_transforms import (
    ResizeLongestSide,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines.transforms import (
    MultipleInputsCompose,
    Pad,
    collate_fn,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_collate_fn():
    """Test collate_fn."""
    batch = [
        {
            "index": 0,
            "images": torch.Tensor([1, 2, 3]),
            "bboxes": [],
            "points": [],
            "gt_masks": [torch.Tensor([1, 2, 3])],
            "original_size": [],
            "padding": [],
            "path": [],
            "labels": [],
        },
        {
            "index": 1,
            "images": torch.Tensor([4, 5, 6]),
            "bboxes": [],
            "points": [],
            "gt_masks": [torch.Tensor([4, 5, 6])],
            "original_size": [],
            "padding": [],
            "path": [],
            "labels": [],
        },
    ]
    expected = {
        "index": [0, 1],
        "images": torch.Tensor([[1, 2, 3], [4, 5, 6]]),
        "bboxes": None,
        "points": None,
        "gt_masks": [torch.Tensor([[1, 2, 3]]), torch.Tensor([[4, 5, 6]])],
        "original_size": [[], []],
        "path": [[], []],
        "labels": [[], []],
        "padding": [[], []],
    }

    results = collate_fn(batch)

    assert results["index"] == expected["index"]
    assert torch.all(results["images"] == expected["images"])
    assert results["bboxes"] == expected["bboxes"]
    assert results["points"] == expected["points"]
    assert len(results["gt_masks"]) == len(expected["gt_masks"])
    for r, e in zip(results["gt_masks"], expected["gt_masks"]):
        assert torch.all(r == e)
    assert results["original_size"] == expected["original_size"]
    assert results["path"] == expected["path"]
    assert results["labels"] == expected["labels"]
    assert results["padding"] == expected["padding"]


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


class TestPad:
    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "item,expected",
        [
            (
                dict(
                    images=torch.zeros((3, 4, 6)),
                    gt_masks=[torch.zeros((4, 6))],
                    bboxes=[[1, 1, 3, 3]],
                    points=[[1, 1, 2, 2]],
                ),
                ((0, 1, 0, 1), (3, 6, 6), [(6, 6)], [[1, 2, 3, 4]], [[1, 2, 2, 3]]),
            ),
            (
                dict(images=torch.zeros((3, 4, 6)), gt_masks=[torch.zeros((4, 6))], bboxes=[[1, 1, 3, 3]], points=None),
                ((0, 1, 0, 1), (3, 6, 6), [(6, 6)], [[1, 2, 3, 4]], None),
            ),
        ],
    )
    def test_call(self, item: Dict[str, Any], expected: Tuple[Any]):
        """Test __call__."""
        pad_transform = Pad()
        expected_padding, expected_images_shape, expected_gt_masks_shape, expected_bboxes, expected_points = expected

        result = pad_transform(item)

        assert result["padding"] == expected_padding
        assert result["images"].shape == expected_images_shape
        assert len(result["gt_masks"]) == len(expected_gt_masks_shape)
        assert all(gt_mask.shape == shape for gt_mask, shape in zip(result["gt_masks"], expected_gt_masks_shape))
        assert result["bboxes"] == expected_bboxes
        assert result["points"] == expected_points


class TestMultipleInputsCompose:
    @e2e_pytest_unit
    def test_call(self):
        """Test __call__."""
        transform1_mock = mock.Mock()
        transform2_mock = mock.Mock(spec=Normalize)

        # Create a sample item
        item = {"images": Tensor([1, 2, 3])}

        # Set the return values of the mocked transforms
        transform1_mock.return_value = item
        transform2_mock.return_value = item["images"]

        # Instantiate the MultipleInputsCompose object with mocked transforms
        transforms = [transform1_mock, transform2_mock]
        multiple_inputs_compose = MultipleInputsCompose(transforms)

        # Call the __call__ method
        results = multiple_inputs_compose(item)

        # Assert that each transform is called appropriately
        transform1_mock.assert_called_once()
        transform2_mock.assert_called_once()

        # Assert the output of the __call__ method
        assert results == {"images": transform2_mock.return_value}
