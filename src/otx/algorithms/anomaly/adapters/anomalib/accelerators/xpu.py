"""Lightning accelerator for XPU device."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, Union

import torch
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.accelerators.accelerator import Accelerator

from otx.algorithms.common.utils.utils import is_xpu_available


class XPUAccelerator(Accelerator):
    """Support for a XPU, optimized for large-scale machine learning."""

    accelerator_name = "xpu"

    def setup_device(self, device: torch.device) -> None:
        """Sets up the specified device."""
        if device.type != "xpu":
            raise RuntimeError(f"Device should be xpu, got {device} instead")

        torch.xpu.set_device(device)

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        """Parses devices for multi-GPU training."""
        if isinstance(devices, list):
            return devices
        return [devices]

    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        """Generates a list of parrallel devices."""
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Returns number of XPU devices available."""
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        """Checks if XPU available."""
        return is_xpu_available()

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Returns XPU devices stats."""
        return {}

    def teardown(self) -> None:
        """Cleans-up XPU-related resources."""
        pass


AcceleratorRegistry.register(
    XPUAccelerator.accelerator_name, XPUAccelerator, description="Accelerator supports XPU devices"
)
