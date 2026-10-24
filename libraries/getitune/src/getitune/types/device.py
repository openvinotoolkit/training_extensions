# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getitune Device type definition."""

from __future__ import annotations

from enum import Enum


class DeviceType(str, Enum):
    """getitune Device type definition."""

    # ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "xpu", "auto")

    auto = "auto"
    gpu = "gpu"
    cpu = "cpu"
    tpu = "tpu"
    ipu = "ipu"
    hpu = "hpu"
    mps = "mps"
    xpu = "xpu"
