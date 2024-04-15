"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import numpy as np


def ensure_rng(rng: int | np.random.RandomState | None = None) -> np.random.RandomState:
    """Coerces input into a random number generator.

    If the input is None, then a global random state is returned.

    If the input is a numeric value, then that is used as a seed to construct a
    random state. Otherwise the input is returned as-is.

    Adapted from [1]_.

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class
    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator

    References:
        .. [1] https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270  # noqa: E501
    """
    if rng is None:
        return np.random.mtrand._rand
    if isinstance(rng, int):
        return np.random.RandomState(rng)
    return rng
