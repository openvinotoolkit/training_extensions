# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""HPO runner and resource manager class."""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import signal
import time
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal

from otx.hpo.hpo_base import HpoBase, Trial, TrialStatus
from otx.hpo.resource_manager import get_resource_manager
from otx.utils import append_signal_handler

if TYPE_CHECKING:
    from collections.abc import Hashable
    from signal import Signals

logger = logging.getLogger(__name__)


@dataclass
class RunningTrial:
    """Data class for a running trial."""

    process: multiprocessing.Process
    trial: Trial
    queue: multiprocessing.Queue


class HpoLoop:
    """HPO loop manager to run trials.

    Args:
        hpo_algo (HpoBase): HPO algorithms.
        train_func (Callable): Function to train a model.
        resource_type (Literal['gpu', 'cpu'], optional): Which type of resource to use.
                                                         If can be changed depending on environment. Defaults to "gpu".
        num_parallel_trial (int | None, optional): How many trials to run in parallel.
                                                   It's used for CPUResourceManager. Defaults to None.
        num_gpu_for_single_trial (int | None, optional): How many GPUs are used for a single trial.
                                                         It's used for GPUResourceManager. Defaults to None.
        available_gpu (str | None, optional): How many GPUs are available. It's used for GPUResourceManager.
                                              Defaults to None.
    """

    def __init__(
        self,
        hpo_algo: HpoBase,
        train_func: Callable,
        resource_type: Literal["gpu", "cpu"] = "gpu",
        num_parallel_trial: int | None = None,
        num_gpu_for_single_trial: int | None = None,
        available_gpu: str | None = None,
    ) -> None:
        self._hpo_algo = hpo_algo
        self._train_func = train_func
        self._running_trials: dict[int, RunningTrial] = {}
        self._mp = multiprocessing.get_context("spawn")
        self._report_queue = self._mp.Queue()
        self._uid_index = 0
        self._trial_fault_count = 0
        self._resource_manager = get_resource_manager(
            resource_type,
            num_parallel_trial,
            num_gpu_for_single_trial,
            available_gpu,
        )
        self._main_pid = os.getpid()

        append_signal_handler(signal.SIGINT, self._terminate_signal_handler)
        append_signal_handler(signal.SIGTERM, self._terminate_signal_handler)

    def run(self) -> None:
        """Run a HPO loop."""
        logger.info("HPO loop starts.")
        try:
            while not self._hpo_algo.is_done() and self._trial_fault_count < 3:
                if self._resource_manager.have_available_resource():
                    trial = self._hpo_algo.get_next_sample()
                    if trial is not None:
                        self._start_trial_process(trial)

                self._remove_finished_process()
                self._get_reports()

                time.sleep(1)
        except Exception as e:
            self._terminate_all_running_processes()
            raise e  # noqa: TRY201
        logger.info("HPO loop is done.")

        if self._trial_fault_count >= 3:
            logger.warning("HPO trials exited abnormally more than three times. HPO is suspended.")

        self._get_reports()
        self._join_all_processes()

    def _start_trial_process(self, trial: Trial) -> None:
        logger.info(f"{trial.id} trial is now running.")
        logger.debug(f"{trial.id} hyper paramter => {trial.configuration}")

        trial.status = TrialStatus.RUNNING
        uid = self._get_uid()

        origin_env = deepcopy(os.environ)
        env = self._resource_manager.reserve_resource(uid)
        if env is not None:
            for key, val in env.items():
                os.environ[key] = val

        trial_queue = self._mp.Queue()
        process = self._mp.Process(
            target=_run_train,
            args=(
                self._train_func,
                trial.get_train_configuration(),
                partial(_report_score, recv_queue=trial_queue, send_queue=self._report_queue, uid=uid),
            ),
        )
        os.environ.clear()
        for key, val in origin_env.items():
            os.environ[key] = val
        self._running_trials[uid] = RunningTrial(process, trial, trial_queue)  # type: ignore[arg-type]
        process.start()

    def _remove_finished_process(self) -> None:
        trial_to_remove = []
        for uid, trial in self._running_trials.items():
            if not trial.process.is_alive():
                if trial.process.exitcode != 0:
                    self._trial_fault_count += 1
                trial.queue.close()
                trial.process.join()
                trial_to_remove.append(uid)

        for uid in trial_to_remove:
            self._running_trials[uid].trial.status = TrialStatus.STOP
            self._resource_manager.release_resource(uid)
            del self._running_trials[uid]

    def _get_reports(self) -> None:
        while not self._report_queue.empty():
            report = self._report_queue.get_nowait()
            trial = self._running_trials[report["uid"]]
            trial_status = self._hpo_algo.report_score(
                report["score"],
                report["progress"],
                trial.trial.id,
                report["done"],
            )
            trial.queue.put_nowait(trial_status)

        self._hpo_algo.save_results()

    def _join_all_processes(self) -> None:
        for val in self._running_trials.values():
            val.queue.close()

        for val in self._running_trials.values():
            val.process.join()

        self._running_trials = {}

    def _get_uid(self) -> int:
        uid = self._uid_index
        self._uid_index += 1
        return uid

    def _terminate_all_running_processes(self) -> None:
        for trial in self._running_trials.values():
            trial.queue.close()
            process = trial.process
            if process.is_alive():
                logger.info(f"Kill child process {process.pid}")
                process.kill()

    def _terminate_signal_handler(self, signum: Signals, frame_) -> None:  # noqa: ANN001
        # This code prevents child processses from being killed unintentionally by proccesses forked from main process
        if self._main_pid != os.getpid():
            return

        self._terminate_all_running_processes()

        singal_name = {2: "SIGINT", 15: "SIGTERM"}
        logger.warning(f"{singal_name[signum]} is sent. process exited.")


def _run_train(train_func: Callable, hp_config: dict, report_func: Callable) -> None:
    # set multi process method as default
    multiprocessing.set_start_method(None, True)
    train_func(hp_config, report_func)


def _report_score(
    score: int | float,
    progress: int | float,
    recv_queue: multiprocessing.Queue,
    send_queue: multiprocessing.Queue,
    uid: Hashable,
    done: bool = False,
) -> TrialStatus:
    logger.debug(f"score : {score}, progress : {progress}, uid : {uid}, pid : {os.getpid()}, done : {done}")
    try:
        send_queue.put_nowait({"score": score, "progress": progress, "uid": uid, "pid": os.getpid(), "done": done})
    except ValueError:
        return TrialStatus.STOP

    try:
        trial_status = recv_queue.get(timeout=3)
    except queue.Empty:
        return TrialStatus.RUNNING

    while not recv_queue.empty():
        trial_status = recv_queue.get_nowait()

    logger.debug(f"trial_status : {trial_status}")
    return trial_status


def run_hpo_loop(
    hpo_algo: HpoBase,
    train_func: Callable,
    resource_type: Literal["gpu", "cpu"] = "gpu",
    num_parallel_trial: int | None = None,
    num_gpu_for_single_trial: int | None = None,
    available_gpu: str | None = None,
) -> None:
    """Run the HPO loop.

    Args:
        hpo_algo (HpoBase): HPO algorithms.
        train_func (Callable): Function to train a model.
        resource_type ('gpu' | 'cpu', optional): Which type of resource to use.
                                                         If can be changed depending on environment. Defaults to "gpu".
        num_parallel_trial (int | None, optional): How many trials to run in parallel.
                                                   It's used for CPUResourceManager. Defaults to None.
        num_gpu_for_single_trial (int | None, optional): How many GPUs are used for a single trial.
                                                         It's used for GPUResourceManager. Defaults to None.
        available_gpu (str | None, optional): How many GPUs are available. It's used for GPUResourceManager.
                                              Defaults to None.
    """
    hpo_loop = HpoLoop(hpo_algo, train_func, resource_type, num_parallel_trial, num_gpu_for_single_trial, available_gpu)
    hpo_loop.run()
