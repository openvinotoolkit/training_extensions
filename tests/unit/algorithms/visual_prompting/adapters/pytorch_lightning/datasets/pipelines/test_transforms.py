"""Tests transforms used for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor
import numpy as np
from torchvision.transforms import Normalize

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
            "images": Tensor([1, 2, 3]),
            "bboxes": np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "points": [],
            "gt_masks": [Tensor([1, 2, 3])],
            "original_size": [],
            "padding": [],
            "path": [],
            "labels": [],
        },
        {
            "index": 1,
            "images": Tensor([4, 5, 6]),
            "bboxes": np.array([[9, 10, 11, 12]]),
            "points": [],
            "gt_masks": [Tensor([4, 5, 6])],
            "original_size": [],
            "padding": [],
            "path": [],
            "labels": [],
        },
    ]
    expected = {
        "index": [0, 1],
        "images": Tensor([[1, 2, 3], [4, 5, 6]]),
        "bboxes": [Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), Tensor([[9, 10, 11, 12]])],
        "points": None,
        "gt_masks": [Tensor([[1, 2, 3]]), Tensor([[4, 5, 6]])],
        "original_size": [[], []],
        "path": [[], []],
        "labels": [[], []],
        "padding": [[], []],
    }

    results = collate_fn(batch)

    assert results["index"] == expected["index"]
    assert torch.all(results["images"] == expected["images"])
    for r, e in zip(results["bboxes"], expected["bboxes"]):
        assert torch.all(r == e)
    assert results["points"] == expected["points"]
    assert len(results["gt_masks"]) == len(expected["gt_masks"])
    for r, e in zip(results["gt_masks"], expected["gt_masks"]):
        assert torch.all(r == e)
    assert results["original_size"] == expected["original_size"]
    assert results["path"] == expected["path"]
    assert results["labels"] == expected["labels"]
    assert results["padding"] == expected["padding"]


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
                ((0, 0, 0, 2), (3, 6, 6), [(4, 6)], [[1, 1, 3, 3]], [[1, 1, 2, 2]]),
            ),
            (
                dict(images=torch.zeros((3, 4, 6)), gt_masks=[torch.zeros((4, 6))], bboxes=[[1, 1, 3, 3]], points=None),
                ((0, 0, 0, 2), (3, 6, 6), [(4, 6)], [[1, 1, 3, 3]], None),
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
    def test_call(self, mocker):
        """Test __call__."""
        transform1_mock = mocker.Mock()
        transform2_mock = mocker.Mock(spec=Normalize)

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
