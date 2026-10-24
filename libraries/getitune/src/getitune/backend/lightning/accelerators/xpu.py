# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning accelerator for XPU device."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.pytorch.accelerators.accelerator import Accelerator
from typing_extensions import override

from getitune.utils.device import is_xpu_available


class XPUAccelerator(Accelerator):
    """Support for a XPU, optimized for large-scale machine learning."""

    accelerator_name = "xpu"

    @override
    def setup_device(self, device: torch.device) -> None:
        """Sets up the specified device."""
        if device.type != "xpu":
            msg = f"Device should be xpu, got {device} instead"
            raise RuntimeError(msg)

        torch.xpu.set_device(device)

    @staticmethod
    @override
    def parse_devices(devices: str | list | torch.device) -> list:
        """Parses devices for multi-GPU training."""
        if isinstance(devices, list):
            return devices
        return [devices]

    @staticmethod
    @override
    def get_parallel_devices(devices: list) -> list[torch.device]:
        """Generates a list of parallel devices."""
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Returns number of XPU devices available."""
        return torch.xpu.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        """Checks if XPU available."""
        return is_xpu_available()

    @staticmethod
    @override
    def name() -> str:
        """The name of the accelerator."""
        return XPUAccelerator.accelerator_name

    def get_device_stats(self, device: str | torch.device) -> dict[str, Any]:
        """Returns XPU devices stats."""
        return {}

    @override
    def teardown(self) -> None:
        """Clean up any state created by the accelerator."""


AcceleratorRegistry.register(
    XPUAccelerator.accelerator_name,
    XPUAccelerator,
    override=True,
    description="Accelerator supports XPU devices",
)
