# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.algo.visual_prompting.utils.postprocess import get_prepadded_size, postprocess_masks
from torch import Tensor


@pytest.mark.parametrize(
    ("input_size", "orig_size", "expected"),
    [
        (6, torch.tensor((8, 8)), torch.Size((8, 8))),
        (6, torch.tensor((10, 8)), torch.Size((10, 8))),
        (6, torch.tensor((8, 10)), torch.Size((8, 10))),
    ],
)
def test_postprocess_masks(input_size: int, orig_size: Tensor, expected: torch.Size) -> None:
    """Test postprocess_masks."""
    masks = torch.zeros((1, 1, 4, 4))

    results = postprocess_masks(masks, input_size, orig_size).squeeze()

    assert results.shape == expected


@pytest.mark.parametrize(
    ("input_image_size", "expected"),
    [
        (torch.tensor((2, 4)), torch.tensor((3, 6))),
        (torch.tensor((4, 2)), torch.tensor((6, 3))),
    ],
)
def test_get_prepadded_size(input_image_size: Tensor, expected: Tensor) -> None:
    """Test get_prepadded_size."""

    longest_side = 6

    results = get_prepadded_size(input_image_size, longest_side)

    assert torch.all(results == expected)
