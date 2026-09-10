# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XPU support for the Hugging Face backend."""

from .xpu import XPUMemoryCallback, clear_xpu_memory

__all__ = ["XPUMemoryCallback", "clear_xpu_memory"]
