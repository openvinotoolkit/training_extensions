"""Lightning accelerator for XPU device."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch
from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.pytorch.accelerators.accelerator import Accelerator
from mmengine.structures import instance_data

from mmcv.ops.nms import NMSop
from mmcv.ops.roi_align import RoIAlign
from otx.algo.detection.utils import monkey_patched_nms, monkey_patched_roi_align
from otx.utils.utils import is_xpu_available


class XPUAccelerator(Accelerator):
    """Support for a XPU, optimized for large-scale machine learning."""

    accelerator_name = "xpu"

    def setup_device(self, device: torch.device) -> None:
        """Sets up the specified device."""
        if device.type != "xpu":
            msg = f"Device should be xpu, got {device} instead"
            raise RuntimeError(msg)

        torch.xpu.set_device(device)
        self.patch_packages_xpu()

    @staticmethod
    def parse_devices(devices: str | list | torch.device) -> list:
        """Parses devices for multi-GPU training."""
        if isinstance(devices, list):
            return devices
        return [devices]

    @staticmethod
    def get_parallel_devices(devices: list) -> list[torch.device]:
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

    def get_device_stats(self, device: str | torch.device) -> dict[str, Any]:
        """Returns XPU devices stats."""
        return {}

    def teardown(self) -> None:
        """Cleans-up XPU-related resources."""
        self.revert_packages_xpu()

    def patch_packages_xpu(self) -> None:
        """Patch packages when xpu is available."""
        # patch instance_data from mmengie
        long_type_tensor = Union[torch.LongTensor, torch.xpu.LongTensor]
        bool_type_tensor = Union[torch.BoolTensor, torch.xpu.BoolTensor]
        instance_data.IndexType = Union[str, slice, int, list, long_type_tensor, bool_type_tensor, np.ndarray]

        # patch nms and roi_align
        self._nms_op_forward = NMSop.forward
        self._roi_align_forward = RoIAlign.forward
        NMSop.forward = monkey_patched_nms
        RoIAlign.forward = monkey_patched_roi_align

    def revert_packages_xpu(self) -> None:
        """Revert packages when xpu is available."""
        NMSop.forward = self._nms_op_forward
        RoIAlign.forward = self._roi_align_forward


AcceleratorRegistry.register(
    XPUAccelerator.accelerator_name,
    XPUAccelerator,
    description="Accelerator supports XPU devices",
)
