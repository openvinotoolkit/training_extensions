# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from copy import deepcopy
from typing import Any, List, Union

import cv2
import numpy as np
import pytest
from PIL import Image

from otx.algorithms.common.adapters.mmcv.pipelines.transforms.augments import (
    Augments,
    CythonAugments,
)


@pytest.fixture
def images() -> List[Image.Image]:
    n_seed = 3003
    n_imgs = 4
    n_shapes = 4
    img_size = 50
    size = [img_size, img_size, 3]

    np.random.seed(n_seed)

    imgs = []
    for _ in range(n_imgs):
        img = np.full(size, 0, dtype=np.uint8)
        for _ in range(n_shapes):
            position = np.random.randint(0, 50, size=[2]).tolist()
            color = np.random.randint(0, 256, size=[3]).tolist()
            marker_type = np.random.randint(0, 7)
            img = cv2.drawMarker(img, position, color, marker_type, thickness=5)
        imgs += [Image.fromarray(img)]

    return imgs


EXACT_EQUAL_TESTS = [
    ("autocontrast", []),
    ("equalize", []),
    ("solarize", [64, 128, 196]),
    ("posterize", [1, 4, 7]),
]


@pytest.mark.parametrize("func,params", EXACT_EQUAL_TESTS)
def test_exact_equal(images: List[Image.Image], func: str, params: List[Any]):
    for img in images:
        for param in params:
            grt = getattr(Augments, func)(deepcopy(img), param)
            tst = getattr(CythonAugments, func)(deepcopy(img), param)

            assert np.array_equal(np.asarray(grt), np.asarray(tst))


APPROX_EQUAL_TESTS = [
    ("color", [0.1, 0.5, 0.9], 1),
    ("contrast", [0.1, 0.5, 0.9], 1),
    ("brightness", [0.1, 0.5, 0.9], 1),
    ("sharpness", [0.25, 0.75, 1.25, 1.75], 1),
    ("rotate", [-35, -15, 15, 35], 1),
    ("shear_x", [-0.8, -0.3, -0.3, 0.8], 1),
    ("shear_y", [-0.8, -0.3, -0.3, 0.8], 1),
    ("translate_x_rel", [-0.8, -0.3, -0.3, 0.8], 1),
    ("translate_y_rel", [-0.8, -0.3, -0.3, 0.8], 1),
]


@pytest.mark.parametrize("func,params,tol", APPROX_EQUAL_TESTS)
def test_approx_equal(images: List[Image.Image], func: str, params: List[Any], tol: Union[float, int]):
    for img in images:
        for param in params:
            grt = getattr(Augments, func)(deepcopy(img), param)
            tst = getattr(CythonAugments, func)(deepcopy(img), param)
            grt = np.array(grt).astype(np.float32)
            tst = np.array(tst).astype(np.float32)
            med = np.median(grt - tst)
            assert med <= tol
