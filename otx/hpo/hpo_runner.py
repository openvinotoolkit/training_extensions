"""HPO runner and resource manager class."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch

from otx.hpo.hpo_base import HpoBase, Trial, TrialStatus
from otx.hpo.utils import check_positive

logger = logging.getLogger(__name__)


class ResourceManager(ABC):
    """Abstract class for resource manager class."""

    @abstractmethod
    def reserve_resource(self, trial_id):
        """Reserve a resource."""
        raise NotImplementedError

    @abstractmethod
    def release_resource(self, trial_id):
        """Release a resource."""
        raise NotImplementedError

    @abstractmethod
    def have_available_resource(self):
        """Check that there is available resource."""
        raise NotImplementedError


class CPUResourceManager(ResourceManager):
    """Resource manager class for CPU.

    Args:
        num_parallel_trial (int, optional): How many trials to run in parallel. Defaults to 4.
    """

    def __init__(self, num_parallel_trial: int = 4):
        check_positive(num_parallel_trial, "num_parallel_trial")

        self._num_parallel_trial = num_parallel_trial
        self._usage_status: List = []

    def reserve_resource(self, trial_id: Any) -> Optional[Dict]:
        """Reserve a resource under 'trial_id'.

        Args:
            trial_id (Any): Name of trial to reserve the resource.

        Raises:
            RuntimeError: If there is already resource reserved by 'trial_id', then raise an error.
        """
        if not self.have_available_resource():
            return None
        if trial_id in self._usage_status:
            raise RuntimeError(f"{trial_id} already has reserved resource.")

        logger.debug(f"{trial_id} reserved.")
        self._usage_status.append(trial_id)
        return {}

    def release_resource(self, trial_id: Any):
        """Release a resource under 'trial_id'.

        Args:
            trial_id (Any): Name of trial which uses the resource to release.
        """
        if trial_id not in self._usage_status:
            logger.warning(f"{trial_id} trial don't use resource now.")
        else:
            self._usage_status.remove(trial_id)
            logger.debug(f"{trial_id} released.")

    def have_available_resource(self):
        """Check that there is available resource."""
        return len(self._usage_status) < self._num_parallel_trial


class GPUResourceManager(ResourceManager):
    """Resource manager class for GPU.

    Args:
        num_gpu_for_single_trial (int, optional): How many GPUs is used for a single trial. Defaults to 1.
        available_gpu (Optional[str], optional): How many GPUs are available. Defaults to None.
    """

    def __init__(self, num_gpu_for_single_trial: int = 1, available_gpu: Optional[str] = None):
        check_positive(num_gpu_for_single_trial, "num_gpu_for_single_trial")

        self._num_gpu_for_single_trial = num_gpu_for_single_trial
        self._available_gpu = self._set_available_gpu(available_gpu)
        self._usage_status: Dict[Any, List] = {}

    def _set_available_gpu(self, available_gpu: Optional[str] = None):
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

    def _transform_gpu_format_from_string_to_arr(self, gpu: str):
        for val in gpu.split(","):
            if not val.isnumeric():
                raise ValueError(
                    "gpu format is wrong. " "gpu should only have numbers delimited by ','.\n" f"your value is {gpu}"
                )
        return [int(val) for val in gpu.split(",")]

    def reserve_resource(self, trial_id: Any) -> Optional[Dict]:
        """Reserve a resource under 'trial_id'.

        Args:
            trial_id (Any): Name of trial to reserve the resource.

        Raises:
            RuntimeError: If there is already resource reserved by 'trial_id', then raise an error.

        Returns:
            Optional[Dict]: Training environment to use.
        """
        if not self.have_available_resource():
            return None
        if trial_id in self._usage_status:
            raise RuntimeError(f"{trial_id} already has reserved resource.")

        resource = list(self._available_gpu[: self._num_gpu_for_single_trial])
        self._available_gpu = self._available_gpu[self._num_gpu_for_single_trial :]

        self._usage_status[trial_id] = resource
        return {"CUDA_VISIBLE_DEVICES": ",".join([str(val) for val in resource])}

    def release_resource(self, trial_id: Any):
        """Release a resource under 'trial_id'.

        Args:
            trial_id (Any): Name of trial which uses the resource to release.
        """
        if trial_id not in self._usage_status:
            logger.warning(f"{trial_id} trial don't use resource now.")
        else:
            self._available_gpu.extend(self._usage_status[trial_id])
            del self._usage_status[trial_id]

    def have_available_resource(self):
        """Check that there is available resource."""
        return len(self._available_gpu) >= self._num_gpu_for_single_trial


def get_resource_manager(
    resource_type: Literal["gpu", "cpu"],
    num_parallel_trial: Optional[int] = None,
    num_gpu_for_single_trial: Optional[int] = None,
    available_gpu: Optional[str] = None,
) -> ResourceManager:
    """Get an appropriate resource manager depending on current environment.

    Args:
        resource_type (Literal["gpu", "cpu"]): Which type of resource to use.
                                               If can be changed depending on environment.
        num_parallel_trial (Optional[int]): How many trials to trun parallelly. It's used for CPUResourceManager.
                                            Defaults to None.
        num_gpu_for_single_trial (Optional[int]): How many GPUs is used for a single trial.
                                                  It's used for GPUResourceManager. Defaults to None.
        available_gpu (Optional[str]): How many GPUs are available. It's used for GPUResourceManager. Defaults to None.

    Raises:
        ValueError: If resource_type is neither 'gpu' nor 'cpu, then raise an error.

    Returns:
        ResourceManager: Resource manager to use.
    """
    if resource_type == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU can't be used now. resource type is modified to cpu.")
        resource_type = "cpu"

    if resource_type == "cpu":
        args = {"num_parallel_trial": num_parallel_trial}
        args = _remove_none_from_dict(args)
        return CPUResourceManager(**args)  # type: ignore
    if resource_type == "gpu":
        args = {"num_gpu_for_single_trial": num_gpu_for_single_trial, "available_gpu": available_gpu}  # type: ignore
        args = _remove_none_from_dict(args)
        return GPUResourceManager(**args)  # type: ignore
    raise ValueError(f"Available resource type is cpu, gpu. Your value is {resource_type}.")


def _remove_none_from_dict(dict_val: Dict):
    key_to_remove = [key for key, val in dict_val.items() if val is None]
    for key in key_to_remove:
        del dict_val[key]
    return dict_val


class HpoLoop:
    """HPO loop manager to run trials.

    Args:
        hpo_algo (HpoBase): HPO algorithms.
        train_func (Callable): Function to train a model.
        resource_type (Literal['gpu', 'cpu'], optional): Which type of resource to use.
                                                         If can be changed depending on environment. Defaults to "gpu".
        num_parallel_trial (Optional[int], optional): How many trials to trun parallelly.
                                                    It's used for CPUResourceManager. Defaults to None.
        num_gpu_for_single_trial (Optional[int], optional): How many GPUs is used for a single trial.
                                                            It's used for GPUResourceManager. Defaults to None.
        available_gpu (Optional[str], optional): How many GPUs are available. It's used for GPUResourceManager.
                                                 Defaults to None.
    """

    def __init__(
        self,
        hpo_algo: HpoBase,
        train_func: Callable,
        resource_type: Literal["gpu", "cpu"] = "gpu",
        num_parallel_trial: Optional[int] = None,
        num_gpu_for_single_trial: Optional[int] = None,
        available_gpu: Optional[str] = None,
    ):
        self._hpo_algo = hpo_algo
        self._train_func = train_func
        self._running_trials: Dict[int, Dict] = {}
        self._mp = multiprocessing.get_context("spawn")
        self._uid_index = 0
        self._resource_manager = get_resource_manager(
            resource_type, num_parallel_trial, num_gpu_for_single_trial, available_gpu
        )

    def run(self):
        """Run a HPO loop."""
        logger.info("HPO loop starts.")
        while not self._hpo_algo.is_done():
            if self._resource_manager.have_available_resource():
                trial = self._hpo_algo.get_next_sample()
                if trial is not None:
                    self._start_trial_process(trial)

            self._remove_finished_process()
            self._get_reports()

        logger.info("HPO loop is done.")
        self._get_reports()
        self._join_all_processes()

    def _start_trial_process(self, trial: Trial):
        logger.info(f"{trial.id} trial is now running.")
        logger.debug(f"{trial.id} hyper paramter => {trial.configuration}")

        trial.status = TrialStatus.RUNNING
        uid = self._get_uid()

        origin_env = os.environ
        env = self._resource_manager.reserve_resource(uid)
        if env is not None:
            for key, val in env.items():
                os.environ[key] = val

        pipe1, pipe2 = self._mp.Pipe(True)
        process = self._mp.Process(
            target=_run_train,
            args=(
                self._train_func,
                trial.get_train_configuration(),
                partial(_report_score, pipe=pipe2, trial_id=trial.id),
            ),
        )
        os.environ = origin_env
        self._running_trials[uid] = {"process": process, "trial": trial, "pipe": pipe1}
        process.start()

    def _remove_finished_process(self):
        trial_to_remove = []
        for uid, val in self._running_trials.items():
            process = val["process"]
            if not process.is_alive():
                val["pipe"].close()
                process.join()
                trial_to_remove.append(uid)

        for uid in trial_to_remove:
            trial = self._running_trials[uid]["trial"]
            trial.status = TrialStatus.STOP
            self._resource_manager.release_resource(uid)
            del self._running_trials[uid]

    def _get_reports(self):
        for trial in self._running_trials.values():
            pipe = trial["pipe"]
            if pipe.poll():
                try:
                    report = pipe.recv()
                except EOFError:
                    continue
                trial_status = self._hpo_algo.report_score(
                    report["score"], report["progress"], report["trial_id"], report["done"]
                )
                pipe.send(trial_status)

        self._hpo_algo.save_results()

    def _join_all_processes(self):
        for val in self._running_trials.values():
            val["pipe"].close()

        for val in self._running_trials.values():
            process = val["process"]
            process.join()

        self._running_trials = {}

    def _get_uid(self) -> int:
        uid = self._uid_index
        self._uid_index += 1
        return uid


def _run_train(train_func: Callable, hp_config: Dict, report_func: Callable):
    # set multi process method as default
    multiprocessing.set_start_method(None, True)  # type: ignore
    train_func(hp_config, report_func)


def _report_score(score: Union[int, float], progress: Union[int, float], pipe, trial_id: Any, done: bool = False):
    logger.debug(f"score : {score}, progress : {progress}, trial_id : {trial_id}, pid : {os.getpid()}, done : {done}")
    try:
        pipe.send({"score": score, "progress": progress, "trial_id": trial_id, "pid": os.getpid(), "done": done})
    except BrokenPipeError:
        return TrialStatus.STOP
    try:
        trial_status = pipe.recv()
    except EOFError:
        return TrialStatus.STOP

    logger.debug(f"trial_status : {trial_status}")
    return trial_status


def run_hpo_loop(
    hpo_algo: HpoBase,
    train_func: Callable,
    resource_type: Literal["gpu", "cpu"] = "gpu",
    num_parallel_trial: Optional[int] = None,
    num_gpu_for_single_trial: Optional[int] = None,
    available_gpu: Optional[str] = None,
):
    """Run the HPO loop.

    Args:
        hpo_algo (HpoBase): HPO algorithms.
        train_func (Callable): Function to train a model.
        resource_type (Literal['gpu', 'cpu'], optional): Which type of resource to use.
                                                         If can be changed depending on environment. Defaults to "gpu".
        num_parallel_trial (Optional[int], optional): How many trials to trun parallelly.
                                                      It's used for CPUResourceManager. Defaults to None.
        num_gpu_for_single_trial (Optional[int], optional): How many GPUs is used for a single trial.
                                                            It's used for GPUResourceManager. Defaults to None.
        available_gpu (Optional[str], optional): How many GPUs are available. It's used for GPUResourceManager.
                                                 Defaults to None.
    """
    hpo_loop = HpoLoop(hpo_algo, train_func, resource_type, num_parallel_trial, num_gpu_for_single_trial, available_gpu)
    hpo_loop.run()
