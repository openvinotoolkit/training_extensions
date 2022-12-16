"""Multi GPU training utility."""

# Copyright (C) 2021 Intel Corporation
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
import os
import signal
import sys
import threading
import time
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from otx.api.configuration import ConfigurableParameters

logger = logging.getLogger(__name__)


def get_gpu_ids(gpus: str) -> List[int]:
    """Get proper GPU indices form `--gpu` arguments.

    Given `--gpus` argument, exclude inappropriate indices and transform to list of int format.

    Args:
        gpus (str): GPU indices to use. Format should be Comma-separated indices.

    Returns:
        List[int]:
            list including proper GPU indices.
    """
    num_available_gpu = torch.cuda.device_count()
    gpu_ids = []
    for gpu_id in gpus.split(","):
        if not gpu_id.isnumeric():
            raise RuntimeError("--gpus argument should be numbers separated by ','.")
        gpu_ids.append(int(gpu_id))

    wrong_gpus = []
    for gpu_idx in gpu_ids:
        if gpu_idx >= num_available_gpu:
            wrong_gpus.append(gpu_idx)

    for wrong_gpu in wrong_gpus:
        gpu_ids.remove(wrong_gpu)

    if wrong_gpus:
        logger.warning(f"Wrong gpu indeces are excluded. {','.join([str(val) for val in gpu_ids])} GPU will be used.")

    return gpu_ids


def set_arguments_to_argv(key: str, value: Optional[str] = None, after_params: bool = False):
    """Add arguments at proper position in `sys.argv`.

    Args:
        key (str): arguement key.
        value (str or None): argument value.
        after_params (bool): whether argument should be after `param` or not.
    """
    if key in sys.argv:
        if value is not None:
            sys.argv[sys.argv.index(key) + 1] = value
    else:
        if not after_params and "params" in sys.argv:
            sys.argv.insert(sys.argv.index("params"), key)
            if value is not None:
                sys.argv.insert(sys.argv.index("params"), value)
        else:
            if after_params and "params" not in sys.argv:
                sys.argv.append("params")
            if value is not None:
                sys.argv.extend([key, value])
            else:
                sys.argv.append(key)


class MultiGPUManager:
    """Class to manage multi GPU training.

    Args:
        train_func (Callable): model training function.
        gpu_ids (str): GPU indices to use. Format should be Comma-separated indices.
        multi_gpu_port (str): port for communication between multi GPU processes.
    """

    def __init__(self, train_func: Callable, gpu_ids: str, multi_gpu_port: str):
        self._train_func = train_func
        self._gpu_ids = get_gpu_ids(gpu_ids)
        self._multi_gpu_port = multi_gpu_port
        self._main_pid = os.getpid()
        self._processes: Optional[List[mp.Process]] = None

    def is_available(self) -> bool:
        """Check multi GPU training is available.

        Returns:
            bool:
                whether multi GPU training is available.
        """
        return len(self._gpu_ids) > 1

    def setup_multi_gpu_train(
        self,
        output_path: str,
        optimized_hyper_parameters: Optional[ConfigurableParameters] = None,
    ):
        """Carry out what should be done to run multi GPU training.

        Args:
            output_path (str): output path where task output are saved.
            optimized_hyper_parameters (ConfigurableParameters or None): hyper parameters reflecting HPO result.
        """
        if optimized_hyper_parameters is not None:  # if HPO is executed, optimized HPs are applied to child processes
            self._set_optimized_hp_for_child_process(optimized_hyper_parameters)

        self._processes = self._spawn_multi_gpu_processes(output_path)

        signal.signal(signal.SIGINT, self._terminate_signal_handler)
        signal.signal(signal.SIGTERM, self._terminate_signal_handler)

        self.initialize_multigpu_train(0, self._gpu_ids, self._multi_gpu_port)

        threading.Thread(target=self._check_child_processes_alive, daemon=True).start()

    def finalize(self):
        """Join all child processes."""
        if self._processes is not None:
            for p in self._processes:
                p.join()

    @staticmethod
    def initialize_multigpu_train(rank: int, gpu_ids: List[int], multi_gpu_port: str):
        """Initilization for multi GPU training.

        Args:
            rank (int): index of multi GPU processes.
            gpu_ids (List[int]): list including which GPU indeces will be used.
            multi_gpu_port (str): port for communication between multi GPU processes.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = multi_gpu_port
        torch.cuda.set_device(gpu_ids[rank])
        dist.init_process_group(backend="nccl", world_size=len(gpu_ids), rank=rank)
        logger.info(f"dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}")

    @staticmethod
    def run_child_process(train_func: Callable, rank: int, gpu_ids: List[int], output_path: str, multi_gpu_port: str):
        """Function for multi GPU child process to execute.

        Args:
            train_func (Callable): model training function.
            rank (int): index of multi GPU processes.
            gpu_ids (List[int]): list including which GPU indeces will be used.
            output_path (str): output path where task output are saved.
            multi_gpu_port (str): port for communication between multi GPU processes.
        """
        gpus_arg_idx = sys.argv.index("--gpus")
        for _ in range(2):
            sys.argv.pop(gpus_arg_idx)
        if "--enable-hpo" in sys.argv:
            sys.argv.remove("--enable-hpo")
        set_arguments_to_argv("--work-dir", output_path)

        MultiGPUManager.initialize_multigpu_train(rank, gpu_ids, multi_gpu_port)

        train_func()

    def _spawn_multi_gpu_processes(self, output_path: str) -> List[mp.Process]:
        processes = []
        spawned_mp = mp.get_context("spawn")
        for rank in range(1, len(self._gpu_ids)):
            task_p = spawned_mp.Process(
                target=MultiGPUManager.run_child_process,
                args=(self._train_func, rank, self._gpu_ids, output_path, self._multi_gpu_port),
            )
            task_p.start()
            processes.append(task_p)

        return processes

    def _terminate_signal_handler(self, signum, _frame):
        # This code prevents child processses from being killed unintentionally by proccesses forked from main process
        if self._main_pid != os.getpid():
            sys.exit()

        self._kill_child_process()

        singal_name = {2: "SIGINT", 15: "SIGTERM"}
        logger.warning(f"{singal_name[signum]} is sent. process exited.")

        sys.exit(1)

    def _kill_child_process(self):
        if self._processes is None:
            return

        for process in self._processes:
            if process.is_alive():
                logger.warning(f"Kill child process {process.pid}")
                process.kill()

    def _set_optimized_hp_for_child_process(self, hyper_parameters: ConfigurableParameters):
        set_arguments_to_argv(
            "--learning_parameters.learning_rate",
            str(hyper_parameters.learning_parameters.learning_rate),  # type: ignore[attr-defined]
            True,
        )
        set_arguments_to_argv(
            "--learning_parameters.batch_size",
            str(hyper_parameters.learning_parameters.batch_size),  # type: ignore[attr-defined]
            True,
        )

    def _check_child_processes_alive(self):
        child_is_running = True
        while child_is_running:
            time.sleep(1)
            for p in self._processes:
                if not p.is_alive() and p.exitcode != 0:
                    child_is_running = False
                    break

        logger.warning("Some of child processes are terminated abnormally. process exits.")
        self._kill_child_process()
        os.kill(self._main_pid, signal.SIGKILL)
