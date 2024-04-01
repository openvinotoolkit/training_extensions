"""Util functions of otx.algorithms.common.adapters.mmdeploy."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections.abc import MutableMapping

import numpy as np


def numpy_2_list(data):
    """Converts NumPy arrays to Python lists."""

    if isinstance(data, np.ndarray):
        return data.tolist()

    if isinstance(data, MutableMapping):
        for key, value in data.items():
            data[key] = numpy_2_list(value)
    elif isinstance(data, (list, tuple)):
        data_ = []
        for value in data:
            data_.append(numpy_2_list(value))
        if isinstance(data, tuple):
            data = tuple(data_)
        else:
            data = data_
    return data
