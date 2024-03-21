# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX utility functions."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Union
import numpy as np
from mmcv.ops.nms import NMSop
from mmcv.ops.roi_align import RoIAlign
from mmengine.structures import instance_data
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy

from otx.algo.detection.utils import monkey_patched_nms, monkey_patched_roi_align

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from otx.core.types.device import DeviceType
    from otx.core.types.task import OTXTaskType


XPU_AVAILABLE = None
try:
    import intel_extension_for_pytorch  # noqa: F401
except ImportError:
    XPU_AVAILABLE = False


def get_using_dot_delimited_key(key: str, target: Any) -> Any:  # noqa: ANN401
    """Get values of attribute in target object using dot delimited key.

    For example, if key is "a.b.c", then get a value of 'target.a.b.c'.
    Target should be object having attributes, dictionary or list.
    To get an element in a list, an integer that is the index of corresponding value can be set as a key.

    Args:
        key (str): dot delimited key.
        val (Any): value to set.
        target (Any): target to set value to.
    """
    splited_key = key.split(".")
    for each_key in splited_key:
        if isinstance(target, dict):
            target = target[each_key]
        elif isinstance(target, list):
            if not each_key.isdigit():
                error_msg = f"Key should be integer but '{each_key}'."
                raise ValueError(error_msg)
            target = target[int(each_key)]
        else:
            target = getattr(target, each_key)
    return target


def set_using_dot_delimited_key(key: str, val: Any, target: Any) -> None:  # noqa: ANN401
    """Set values to attribute in target object using dot delimited key.

    For example, if key is "a.b.c", then value is set at 'target.a.b.c'.
    Target should be object having attributes, dictionary or list.
    To get an element in a list, an integer that is the index of corresponding value can be set as a key.

    Args:
        key (str): dot delimited key.
        val (Any): value to set.
        target (Any): target to set value to.
    """
    splited_key = key.split(".")
    for each_key in splited_key[:-1]:
        if isinstance(target, dict):
            target = target[each_key]
        elif isinstance(target, list):
            if not each_key.isdigit():
                error_msg = f"Key should be integer but '{each_key}'."
                raise ValueError(error_msg)
            target = target[int(each_key)]
        else:
            target = getattr(target, each_key)

    if isinstance(target, dict):
        target[splited_key[-1]] = val
    elif isinstance(target, list):
        if not splited_key[-1].isdigit():
            error_msg = f"Key should be integer but '{splited_key[-1]}'."
            raise ValueError(error_msg)
        target[int(splited_key[-1])] = val
    else:
        setattr(target, splited_key[-1], val)


def get_decimal_point(num: int | float) -> int:
    """Find a decimal point from the given float.

    Args:
        num (int | float): float to find a decimal point from.

    Returns:
        int: decimal point.
    """
    if isinstance((exponent := Decimal(str(num)).as_tuple().exponent), int):
        return abs(exponent)
    error_msg = f"Can't get an exponent from {num}."
    raise ValueError(error_msg)


def find_file_recursively(directory: Path, file_name: str) -> Path | None:
    """Find the file from the direcotry recursively. If multiple files have a same name, return one of them.

    Args:
        directory (Path): directory where to find.
        file_name (str): file name to find.

    Returns:
        Path | None: Found file. If it's failed to find a file, return None.
    """
    if found_file := list(directory.rglob(file_name)):
        return found_file[0]
    return None


def remove_matched_files(directory: Path, pattern: str, file_to_leave: Path | None = None) -> None:
    """Remove all files matched to pattern except file_to_leave.

    Args:
        directory (Path): direcetory to find files to remove.
        pattern (str): pattern to match a file name.
        file_not_to_remove (Path | None, optional): files to leave. Defaults to None.
    """
    for weight in directory.rglob(pattern):
        if weight != file_to_leave:
            weight.unlink()


def is_xpu_available() -> bool:
    """Checks if XPU device is available."""
    global XPU_AVAILABLE  # noqa: PLW0603
    if XPU_AVAILABLE is None:
        XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
    return XPU_AVAILABLE


def patch_packages_xpu(task: str | OTXTaskType, accelerator: str | DeviceType) -> None:
    """Patch packages when xpu is available."""
    import lightning.pytorch as pl

    def patched_setup_optimizers(self, trainer: pl.Trainer) -> None:
        """Sets up optimizers."""
        super(SingleDeviceStrategy).setup_optimizers(trainer)
        if len(self.optimizers) != 1:  # type: ignore[has-type]
            msg = "XPU strategy doesn't support multiple optimizers"
            raise RuntimeError(msg)
        model, optimizer = torch.xpu.optimize(trainer.model, optimizer=self.optimizers[0])  # type: ignore[has-type]
        self.optimizers = [optimizer]
        self.model = model

    # patch instance_data from mmengie
    long_type_tensor = Union[torch.LongTensor, torch.xpu.LongTensor]
    bool_type_tensor = Union[torch.BoolTensor, torch.xpu.BoolTensor]
    instance_data.IndexType = Union[str, slice, int, list, long_type_tensor, bool_type_tensor, np.ndarray]

    # patch nms, roi_align and setup_optimizers for the lightning strategy
    global _nms_op_forward, _roi_align_forward, _setup_optimizers
    _nms_op_forward = NMSop.forward
    _roi_align_forward = RoIAlign.forward
    _setup_optimizers = SingleDeviceStrategy.setup_optimizers
    NMSop.forward = monkey_patched_nms
    RoIAlign.forward = monkey_patched_roi_align
    SingleDeviceStrategy.setup_optimizers = patched_setup_optimizers


def revert_packages_xpu():
    """Revert packages when xpu is available."""
    NMSop.forward = _nms_op_forward
    RoIAlign.forward = _roi_align_forward
    SingleDeviceStrategy.setup_optimizers = _setup_optimizers
