# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XPU-specific support for the Hugging Face backend."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback

from getitune.utils.device import is_xpu_available

__all__ = ["XPUMemoryCallback", "clear_xpu_memory"]


def clear_xpu_memory() -> None:
    """Release cached XPU allocations. A no-op when XPU isn't available."""
    if not is_xpu_available():
        return
    import torch

    torch.xpu.empty_cache()


class XPUMemoryCallback(TrainerCallback):
    """Clears the XPU allocator cache at the end of every epoch."""

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Clear the XPU cache."""
        clear_xpu_memory()
