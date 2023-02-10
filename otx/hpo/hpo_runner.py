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
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Union

from otx.hpo.hpo_base import HpoBase, Trial, TrialStatus
from otx.hpo.resource_manager import get_resource_manager

logger = logging.getLogger(__name__)


class HpoLoop:
    """HPO loop manager to run trials.

    Args:
        hpo_algo (HpoBase): HPO algorithms.
        train_func (Callable): Function to train a model.
        resource_type (Literal['gpu', 'cpu'], optional): Which type of resource to use.
                                                         If can be changed depending on environment. Defaults to "gpu".
        num_parallel_trial (Optional[int], optional): How many trials to run in parallel.
                                                    It's used for CPUResourceManager. Defaults to None.
        num_gpu_for_single_trial (Optional[int], optional): How many GPUs are used for a single trial.
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

        origin_env = deepcopy(os.environ)
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
        num_parallel_trial (Optional[int], optional): How many trials to run in parallel.
                                                      It's used for CPUResourceManager. Defaults to None.
        num_gpu_for_single_trial (Optional[int], optional): How many GPUs are used for a single trial.
                                                            It's used for GPUResourceManager. Defaults to None.
        available_gpu (Optional[str], optional): How many GPUs are available. It's used for GPUResourceManager.
                                                 Defaults to None.
    """
    hpo_loop = HpoLoop(hpo_algo, train_func, resource_type, num_parallel_trial, num_gpu_for_single_trial, available_gpu)
    hpo_loop.run()
