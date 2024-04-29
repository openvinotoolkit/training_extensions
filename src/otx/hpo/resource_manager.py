# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Resource manager class for HPO runner."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import torch

from otx.core.types.device import DeviceType
from otx.hpo.utils import check_positive
from otx.utils.utils import is_xpu_available

if TYPE_CHECKING:
    from collections.abc import Hashable

logger = logging.getLogger(__name__)


class BaseResourceManager(ABC):
    """Abstract class for resource manager class."""

    @abstractmethod
    def reserve_resource(self, trial_id: Hashable) -> dict | None:
        """Reserve a resource."""
        raise NotImplementedError

    @abstractmethod
    def release_resource(self, trial_id: Hashable) -> None:
        """Release a resource."""
        raise NotImplementedError

    @abstractmethod
    def have_available_resource(self) -> bool:
        """Check that there is available resource."""
        raise NotImplementedError


class CPUResourceManager(BaseResourceManager):
    """Resource manager class for CPU.

    Args:
        num_parallel_trial (int, optional): How many trials to run in parallel. Defaults to 1.
    """

    def __init__(self, num_parallel_trial: int = 1) -> None:
        check_positive(num_parallel_trial, "num_parallel_trial")

        self._num_parallel_trial = num_parallel_trial
        self._usage_status: list[Any] = []

    def reserve_resource(self, trial_id: Hashable) -> dict | None:
        """Reserve a resource under 'trial_id'.

        Args:
            trial_id (Any): Name of trial to reserve the resource.

        Raises:
            RuntimeError: If there is already resource reserved by 'trial_id', then raise an error.
        """
        if not self.have_available_resource():
            return None
        if trial_id in self._usage_status:
            error_msg = f"{trial_id} already has reserved resource."
            raise RuntimeError(error_msg)

        logger.debug(f"{trial_id} reserved.")
        self._usage_status.append(trial_id)
        return {}

    def release_resource(self, trial_id: Hashable) -> None:
        """Release a resource under 'trial_id'.

        Args:
            trial_id (Any): Name of trial which uses the resource to release.
        """
        if trial_id not in self._usage_status:
            logger.warning(f"{trial_id} trial don't use resource now.")
        else:
            self._usage_status.remove(trial_id)
            logger.debug(f"{trial_id} released.")

    def have_available_resource(self) -> bool:
        """Check that there is available resource."""
        return len(self._usage_status) < self._num_parallel_trial


class AcceleratorManager(BaseResourceManager):
    """Abstract Resource manager class for accelerators.

    Args:
        num_devices_per_trial (int, optional): Number of devices used for a single trial. Defaults to 1.
        num_parallel_trial (int | None, optional): How many trials to run in parallel. Defaults to None.
    """

    def __init__(self, num_devices_per_trial: int = 1, num_parallel_trial: int | None = None) -> None:
        check_positive(num_devices_per_trial, "num_devices_per_trial")
        if num_parallel_trial is not None:
            check_positive(num_parallel_trial, "num_parallel_trial")

        self._num_devices_per_trial = num_devices_per_trial
        self._available_devices = self._get_available_devices(num_parallel_trial)
        self._usage_status: dict[Any, list] = {}

    @abstractmethod
    def _get_available_devices(self, num_parallel_trial: int | None = None) -> list[int]:
        raise NotImplementedError

    def reserve_resource(self, trial_id: Hashable) -> dict[str, str] | None:
        """Reserve a resource under 'trial_id'.

        Args:
            trial_id (Hashable): Name of trial to reserve the resource.

        Raises:
            RuntimeError: If there is already resource reserved by 'trial_id', then raise an error.

        Returns:
            dict[str, str] | None: Training environment to use.
        """
        if not self.have_available_resource():
            return None
        if trial_id in self._usage_status:
            msg = f"{trial_id} already has reserved resource."
            raise RuntimeError(msg)

        resource = list(self._available_devices[: self._num_devices_per_trial])
        self._available_devices = self._available_devices[self._num_devices_per_trial :]

        self._usage_status[trial_id] = resource
        return self._make_env_var_for_train(resource)

    @abstractmethod
    def _make_env_var_for_train(self, device_arr: list[int]) -> dict[str, str]:
        raise NotImplementedError

    def release_resource(self, trial_id: Hashable) -> None:
        """Release a resource under 'trial_id'.

        Args:
            trial_id (Hashable): Name of trial which uses the resource to release.
        """
        if trial_id not in self._usage_status:
            logger.warning(f"{trial_id} trial don't use resource now.")
        else:
            self._available_devices.extend(self._usage_status[trial_id])
            del self._usage_status[trial_id]

    def have_available_resource(self) -> bool:
        """Check that there is available resource."""
        return len(self._available_devices) >= self._num_devices_per_trial


class GPUResourceManager(AcceleratorManager):
    """Resource manager class for GPU."""

    def _get_available_devices(self, num_parallel_trial: int | None = None) -> list[int]:
        if (cuda_visible_devices := os.getenv("CUDA_VISIBLE_DEVICES")) is not None:
            available_gpu_arr = _cvt_comma_delimited_str_to_list(cuda_visible_devices)
        else:
            available_gpu_arr = list(range(torch.cuda.device_count()))
        if num_parallel_trial is not None:
            available_gpu_arr = available_gpu_arr[: num_parallel_trial * self._num_devices_per_trial]

        return available_gpu_arr

    def _make_env_var_for_train(self, device_arr: list[int]) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": ",".join([str(val) for val in device_arr])}


class XPUResourceManager(AcceleratorManager):
    """Resource manager class for XPU."""

    def _get_available_devices(self, num_parallel_trial: int | None = None) -> list[int]:
        visible_devices = os.getenv("ONEAPI_DEVICE_SELECTOR")
        if isinstance(visible_devices, str) and "level_zero:" in visible_devices:
            available_devices_arr = _cvt_comma_delimited_str_to_list(visible_devices.split("level_zero:")[1])
        else:
            available_devices_arr = list(range(torch.xpu.device_count()))
        if num_parallel_trial is not None:
            available_devices_arr = available_devices_arr[: num_parallel_trial * self._num_devices_per_trial]

        return available_devices_arr

    def _make_env_var_for_train(self, device_arr: list[int]) -> dict[str, str]:
        return {"ONEAPI_DEVICE_SELECTOR": "level_zero:" + ",".join([str(val) for val in device_arr])}


def get_resource_manager(
    resource_type: Literal[DeviceType.cpu, DeviceType.gpu, DeviceType.xpu],
    num_parallel_trial: int | None = None,
    num_devices_per_trial: int = 1,
) -> BaseResourceManager:
    """Get an appropriate resource manager depending on current environment.

    Args:
        resource_type (Literal[DeviceType.cpu, DeviceType.gpu, DeviceType.xpu]):
            Which type of resource to use. It can be changed depending on environment.
        num_parallel_trial (int, optional): How many trials to run in parallel. Defaults to None.
        num_devices_per_trial (int, optinal): How many accelerators is used for a single trial.
                                              It's used for AcceleratorManager. Defaults to 1.

    Raises:
        ValueError: If resource_type is neither 'cpu', 'gpu' nor 'xpu', then raise an error.

    Returns:
        BaseResourceManager: Resource manager to use.
    """
    if (resource_type == DeviceType.gpu and not torch.cuda.is_available()) or (
        resource_type == DeviceType.xpu and not is_xpu_available()
    ):
        logger.warning(f"{resource_type} can't be used now. resource type is modified to cpu.")
        resource_type = DeviceType.cpu

    if resource_type == DeviceType.cpu:
        args = {"num_parallel_trial": num_parallel_trial}
        args = _remove_none_from_dict(args)
        return CPUResourceManager(**args)  # type: ignore[arg-type]
    if resource_type == DeviceType.gpu:
        args = {"num_devices_per_trial": num_devices_per_trial, "num_parallel_trial": num_parallel_trial}  # type: ignore[dict-item]
        args = _remove_none_from_dict(args)
        return GPUResourceManager(**args)  # type: ignore[arg-type]
    if resource_type == DeviceType.xpu:
        args = {"num_devices_per_trial": num_devices_per_trial, "num_parallel_trial": num_parallel_trial}  # type: ignore[dict-item]
        args = _remove_none_from_dict(args)
        return XPUResourceManager(**args)  # type: ignore[arg-type]
    msg = f"Available resource type is cpu, gpu or xpu. Your value is {resource_type}."
    raise ValueError(msg)


def _remove_none_from_dict(dict_val: dict) -> dict:
    key_to_remove = [key for key, val in dict_val.items() if val is None]
    for key in key_to_remove:
        del dict_val[key]
    return dict_val


def _cvt_comma_delimited_str_to_list(string: str) -> list[int]:
    for val in string.split(","):
        if not val.isnumeric():
            msg = f"Wrong format is given. String should have numbers delimited by ','.\nyour value is {string}"
            raise ValueError(msg)
    return [int(val) for val in string.split(",")]
