# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Resource manager class for HPO runner."""

from __future__ import annotations

import logging
import os
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import torch

from otx.hpo.utils import check_positive

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
        num_parallel_trial (int, optional): How many trials to run in parallel. Defaults to 4.
    """

    def __init__(self, num_parallel_trial: int = 4) -> None:
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


class GPUResourceManager(BaseResourceManager):
    """Resource manager class for GPU.

    Args:
        num_gpu_for_single_trial (int, optional): How many GPUs is used for a single trial. Defaults to 1.
        available_gpu (str | None, optional): How many GPUs are available. Defaults to None.
    """

    def __init__(self, num_gpu_for_single_trial: int = 1, available_gpu: str | None = None) -> None:
        check_positive(num_gpu_for_single_trial, "num_gpu_for_single_trial")

        self._num_gpu_for_single_trial = num_gpu_for_single_trial
        self._available_gpu = self._set_available_gpu(available_gpu)
        self._usage_status: dict[Any, list] = {}

    def _set_available_gpu(self, available_gpu: str | None = None) -> list[int]:
        if available_gpu is None:
            cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is not None:
                available_gpu_arr = self._transform_gpu_format_from_string_to_arr(cuda_visible_devices)
            else:
                num_gpus = torch.cuda.device_count()
                available_gpu_arr = list(range(num_gpus))
        else:
            available_gpu_arr = self._transform_gpu_format_from_string_to_arr(available_gpu)

        return available_gpu_arr

    def _transform_gpu_format_from_string_to_arr(self, gpu: str) -> list[int]:
        for val in gpu.split(","):
            if not val.isnumeric():
                error_msg = f"gpu format is wrong. gpu should only have numbers delimited by ','.\nyour value is {gpu}"
                raise ValueError(error_msg)
        return [int(val) for val in gpu.split(",")]

    def reserve_resource(self, trial_id: Hashable) -> dict | None:
        """Reserve a resource under 'trial_id'.

        Args:
            trial_id (Any): Name of trial to reserve the resource.

        Raises:
            RuntimeError: If there is already resource reserved by 'trial_id', then raise an error.

        Returns:
            dict | None: Training environment to use.
        """
        if not self.have_available_resource():
            return None
        if trial_id in self._usage_status:
            error_msg = f"{trial_id} already has reserved resource."
            raise RuntimeError(error_msg)

        resource = list(self._available_gpu[: self._num_gpu_for_single_trial])
        self._available_gpu = self._available_gpu[self._num_gpu_for_single_trial :]

        self._usage_status[trial_id] = resource
        return {"CUDA_VISIBLE_DEVICES": ",".join([str(val) for val in resource])}

    def release_resource(self, trial_id: Hashable) -> None:
        """Release a resource under 'trial_id'.

        Args:
            trial_id (Any): Name of trial which uses the resource to release.
        """
        if trial_id not in self._usage_status:
            logger.warning(f"{trial_id} trial don't use resource now.")
        else:
            self._available_gpu.extend(self._usage_status[trial_id])
            del self._usage_status[trial_id]

    def have_available_resource(self) -> bool:
        """Check that there is available resource."""
        return len(self._available_gpu) >= self._num_gpu_for_single_trial


def get_resource_manager(
    resource_type: Literal["gpu", "cpu"],
    num_parallel_trial: int | None = None,
    num_gpu_for_single_trial: int | None = None,
    available_gpu: str | None = None,
) -> BaseResourceManager:
    """Get an appropriate resource manager depending on current environment.

    Args:
        resource_type (Literal["gpu", "cpu"]): Which type of resource to use.
                                               If can be changed depending on environment.
        num_parallel_trial (int | None, optional): How many trials to run in parallel. It's used for CPUResourceManager.
                                            Defaults to None.
        num_gpu_for_single_trial (int | None, optional): How many GPUs is used for a single trial.
                                                         It's used for GPUResourceManager. Defaults to None.
        available_gpu (str | None, optional): How many GPUs are available. It's used for GPUResourceManager.
                                              Defaults to None.

    Raises:
        ValueError: If resource_type is neither 'gpu' nor 'cpu', then raise an error.

    Returns:
        BaseResourceManager: Resource manager to use.
    """
    if resource_type == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU can't be used now. resource type is modified to cpu.")
        resource_type = "cpu"

    if resource_type == "cpu":
        args = {"num_parallel_trial": num_parallel_trial}
        args = _remove_none_from_dict(args)
        return CPUResourceManager(**args)  # type: ignore[arg-type]
    if resource_type == "gpu":
        args = {"num_gpu_for_single_trial": num_gpu_for_single_trial, "available_gpu": available_gpu}  # type: ignore[dict-item]
        args = _remove_none_from_dict(args)
        return GPUResourceManager(**args)  # type: ignore[arg-type]
    error_msg = f"Available resource type is cpu, gpu. Your value is {resource_type}."
    raise ValueError(error_msg)


def _remove_none_from_dict(dict_val: dict) -> dict:
    key_to_remove = [key for key, val in dict_val.items() if val is None]
    for key in key_to_remove:
        del dict_val[key]
    return dict_val
