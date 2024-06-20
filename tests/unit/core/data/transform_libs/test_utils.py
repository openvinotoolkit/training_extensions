# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch
from otx.core.data.transform_libs.utils import get_image_shape, to_np_image
from torch import Tensor


@pytest.mark.parametrize(("img", "expected_shape"), [(np.zeros((1, 2, 3)), (1, 2)), (torch.zeros((1, 2, 3)), (2, 3))])
@pytest.mark.parametrize("is_list", [True, False])
def test_get_image_shape(img: np.ndarray | Tensor | list, is_list: bool, expected_shape: tuple[int, int]) -> None:
    if is_list:
        img = [img, img]

    results = get_image_shape(img)

    assert results == expected_shape


@pytest.mark.parametrize("img", [np.zeros((1, 2, 3)), torch.zeros((1, 2, 3))])
@pytest.mark.parametrize("is_list", [True, False])
def test_to_np_image(img: np.ndarray | Tensor | list, is_list: bool) -> None:
    results = to_np_image(img)

    if is_list:
        assert all(isinstance(r, np.ndarray) for r in results)
    else:
        assert isinstance(results, np.ndarray)
