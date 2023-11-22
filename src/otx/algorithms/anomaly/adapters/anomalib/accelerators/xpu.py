# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, Union

import torch
from pytorch_lightning.accelerators.accelerator import Accelerator
from otx.algorithms.common.utils import is_xpu_available


class XPUAccelerator(Accelerator):
    """Support for a hypothetical XPU, optimized for large-scale machine learning."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        print(device)
        device = torch.device("xpu", 0)
        if device.type != "xpu":
            raise RuntimeError(f"Device should be XPU, got {device} instead")

        torch.xpu.set_device(device)

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument
        return [devices]

    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return torch.xpu.device_count()

    def teardown(self) -> None:
        pass

    @staticmethod
    def is_available() -> bool:
        return is_xpu_available()

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        # Return optional device statistics for loggers
        return {}

    @classmethod
    def register_accelerators(cls, accelerator_registry):
        accelerator_registry.register(
            "xpu",
            cls,
            description=f"XPU accelerator supports Intel ARC and Max GPUs.",
        )