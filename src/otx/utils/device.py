# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX device utility functions."""

from __future__ import annotations

from typing import Literal

import torch

XPU_AVAILABLE = None


def is_xpu_available() -> bool:
    """Checks if XPU device is available."""
    global XPU_AVAILABLE  # noqa: PLW0603
    if XPU_AVAILABLE is None:
        XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
    return XPU_AVAILABLE


def get_available_device() -> Literal["xpu", "cuda", "cpu"]:
    """Returns an available device in the order of xpu, cuda, and cpu."""
    if is_xpu_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
