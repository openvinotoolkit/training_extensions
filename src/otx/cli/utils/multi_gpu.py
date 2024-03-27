"""Multi GPU training utility."""

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

import datetime
import os
import signal
import socket
import sys
import threading
import time
from contextlib import closing
from typing import Callable, List, Optional, Union

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from otx.api.configuration import ConfigurableParameters
from otx.utils.logger import get_logger

logger = get_logger()


def _get_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


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
            raise ValueError("--gpus argument should be numbers separated by ','.")
        gpu_ids.append(int(gpu_id))

    wrong_gpus = []
    for gpu_idx in gpu_ids:
        if gpu_idx >= num_available_gpu:
            wrong_gpus.append(gpu_idx)

    for wrong_gpu in wrong_gpus:
        gpu_ids.remove(wrong_gpu)

    if wrong_gpus:
        logger.warning(f"Wrong gpu indices are excluded. {','.join([str(val) for val in gpu_ids])} GPU will be used.")

    return gpu_ids


def set_arguments_to_argv(keys: Union[str, List[str]], value: Optional[str] = None, after_params: bool = False):
    """Add arguments at proper position in `sys.argv`.

    Args:
        keys (str or List[str]): arguement keys.
        value (str or None): argument value.
        after_params (bool): whether argument should be after `param` or not.
    """
    if not isinstance(keys, list):
        keys = [keys]
    for key in keys:
        if key in sys.argv:
            if value is not None:
                sys.argv[sys.argv.index(key) + 1] = value
            return

    key = keys[0]
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


def is_multigpu_child_process():
    """Check current process is a child process for multi GPU training."""
    return (dist.is_initialized() or "TORCHELASTIC_RUN_ID" in os.environ) and os.environ["LOCAL_RANK"] != "0"


class MultiGPUManager:
    """Class to manage multi GPU training.

    Args:
        train_func (Callable): model training function.
        gpu_ids (str): GPU indices to use. Format should be Comma-separated indices.
        rdzv_endpoint (str): Rendezvous endpoint for multi-node training.
        base_rank (int): Base rank of the worker.
        world_size (int): Total number of workers in a worker group.
        start_time (Optional[datetime.datetime]): Time when process starts.
            This value is used to decide timeout argument of distributed training.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        train_func: Callable,
        gpu_ids: str,
        rdzv_endpoint: str = "localhost:0",
        base_rank: int = 0,
        world_size: int = 0,
        start_time: Optional[datetime.datetime] = None,
    ):
        if ":" not in rdzv_endpoint:
            raise ValueError("rdzv_endpoint must be in form <host>:<port>.")
        host, port = rdzv_endpoint.split(":")
        if port == "0":
            assert host in ["localhost", "127.0.0.1"]
            port = _get_free_port()
        rdzv_endpoint = f"{host}:{port}"

        self._train_func = train_func
        self._gpu_ids = get_gpu_ids(gpu_ids)
        self._rdzv_endpoint = rdzv_endpoint
        self._base_rank = base_rank
        if world_size == 0:
            world_size = len(self._gpu_ids)
        self._world_size = world_size
        self._main_pid = os.getpid()
        self._processes: List[mp.Process] = []

        if start_time is not None:
            elapsed_time = datetime.datetime.now() - start_time
            if elapsed_time > datetime.timedelta(seconds=40):
                os.environ["TORCH_DIST_TIMEOUT"] = str(int(elapsed_time.total_seconds() * 1.5))

    def is_available(self) -> bool:
        """Check multi GPU training is available.

        Returns:
            bool:
                whether multi GPU training is available.
        """
        return (
            len(self._gpu_ids) > 1
            and "TORCHELASTIC_RUN_ID"
            not in os.environ  # If otx is executed by torchrun, then otx multi gpu interface is disabled.
        )

    def setup_multi_gpu_train(
        self,
        output_path: str,
        optimized_hyper_parameters: Optional[ConfigurableParameters] = None,
    ):
        """Carry out what should be done to run multi GPU training.

        Args:
            output_path (str): output path where task output are saved.
            optimized_hyper_parameters (ConfigurableParameters or None): hyper parameters reflecting HPO result.

        Returns:
            str:
                If output_path is None, make a temporary directory and return it.
        """
        if optimized_hyper_parameters is not None:  # if HPO is executed, optimized HPs are applied to child processes
            self._set_optimized_hp_for_child_process(optimized_hyper_parameters)

        self._processes = self._spawn_multi_gpu_processes(output_path)

        signal.signal(signal.SIGINT, self._terminate_signal_handler)
        signal.signal(signal.SIGTERM, self._terminate_signal_handler)

        self.initialize_multigpu_train(self._rdzv_endpoint, self._base_rank, 0, self._gpu_ids, self._world_size)

        threading.Thread(target=self._check_child_processes_alive, daemon=True).start()

    def finalize(self):
        """Join all child processes."""
        for p in self._processes:
            if p.join(30) is None and p.exitcode is None:
                p.kill()

    @staticmethod
    def initialize_multigpu_train(
        rdzv_endpoint: str,
        rank: int,
        local_rank: int,
        gpu_ids: List[int],
        world_size: int,
    ):
        """Initilization for multi GPU training.

        Args:
            rdzv_endpoint (str): Rendezvous endpoint for multi-node training.
            rank (int): The rank of worker within a worker group.
            local_rank (int): The rank of worker within a local worker group.
            gpu_ids (List[int]): list including which GPU indeces will be used.
            world_size (int): Total number of workers in a worker group.
        """

        host, port = rdzv_endpoint.split(":")
        os.environ["MASTER_ADDR"] = host
        os.environ["MASTER_PORT"] = port
        os.environ["LOCAL_WORLD_SIZE"] = str(len(gpu_ids))
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)

    @staticmethod
    def run_child_process(
        train_func: Callable,
        output_path: str,
        rdzv_endpoint: str,
        rank: int,
        local_rank: int,
        gpu_ids: List[int],
        world_size: int,
    ):
        """Function for multi GPU child process to execute.

        Args:
            train_func (Callable): model training function.
            output_path (str): output path where task output are saved.
            rdzv_endpoint (str): Rendezvous endpoint for multi-node training.
            rank (int): The rank of worker within a worker group.
            local_rank (int): The rank of worker within a local worker group.
            gpu_ids (List[int]): list including which GPU indeces will be used.
            world_size (int): Total number of workers in a worker group.
        """

        # initialize start method
        mp.set_start_method(method=None, force=True)

        gpus_arg_idx = sys.argv.index("--gpus")
        for _ in range(2):
            sys.argv.pop(gpus_arg_idx)
        if "--enable-hpo" in sys.argv:
            sys.argv.remove("--enable-hpo")
        set_arguments_to_argv(["-o", "--output"], output_path)
        set_arguments_to_argv("--rdzv-endpoint", rdzv_endpoint)

        MultiGPUManager.initialize_multigpu_train(rdzv_endpoint, rank, local_rank, gpu_ids, world_size)

        threading.Thread(target=MultiGPUManager.check_parent_processes_alive, daemon=True).start()

        train_func()

    @staticmethod
    def check_parent_processes_alive():
        """Check parent process is alive and if not, exit by itself."""
        cur_process = psutil.Process()
        parent = cur_process.parent()
        while True:
            time.sleep(1)
            if not parent.is_running():
                break

        logger.warning("Parent process is terminated abnormally. Process exits.")
        cur_process.kill()

    def _spawn_multi_gpu_processes(self, output_path: str) -> List[mp.Process]:
        processes = []
        ctx = mp.get_context("spawn")

        # set CUDA_VISIBLE_DEVICES to make child process use proper GPU
        origin_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if origin_cuda_visible_devices is not None:
            cuda_visible_devices = origin_cuda_visible_devices.split(",")
        else:
            cuda_visible_devices = [str(i) for i in range(torch.cuda.device_count())]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([cuda_visible_devices[gpu_idx] for gpu_idx in self._gpu_ids])

        for rank in range(1, len(self._gpu_ids)):
            task_p = ctx.Process(
                target=MultiGPUManager.run_child_process,
                args=(
                    self._train_func,
                    output_path,
                    self._rdzv_endpoint,
                    self._base_rank + rank,
                    rank,
                    self._gpu_ids,
                    self._world_size,
                ),
            )
            task_p.start()
            processes.append(task_p)

        if origin_cuda_visible_devices is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = origin_cuda_visible_devices

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
