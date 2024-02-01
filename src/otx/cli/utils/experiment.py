"""Utils function for experiments."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Union, no_type_check

import psutil
import yaml

from otx.utils.logger import get_logger

try:
    import pynvml
except ImportError:
    pynvml = None

logger = get_logger()
GIB = 1024**3
AVAILABLE_RESOURCE_TYPE = ["cpu", "gpu"]


class ResourceTracker:
    """Class to track resources usage.

    Args:
        output_path (Union[str, Path]): Output file path to save CPU & GPU utilization and max meory usage values.
        resource_type (str, optional): Which resource to track. Available values are cpu, gpu or all now.
                                        Defaults to "all".
        gpu_ids (Optional[str]): GPU indices to record.
    """

    def __init__(self, output_path: Union[str, Path], resource_type: str = "all", gpu_ids: Optional[str] = None):
        if isinstance(output_path, str):
            output_path = Path(output_path)
        self.output_path = output_path
        if resource_type == "all":
            self._resource_type = AVAILABLE_RESOURCE_TYPE
        else:
            self._resource_type = [val for val in resource_type.split(",")]

        gpu_ids_arr = None
        if gpu_ids is not None:
            gpu_ids_arr = [int(idx) for idx in gpu_ids.split(",")]
            gpu_ids_arr[0] = 0

        self._gpu_ids: Union[List[int], None] = gpu_ids_arr
        self._mem_check_proc: Union[mp.Process, None] = None
        self._queue: Union[mp.Queue, None] = None

    def start(self):
        """Run a process which tracks resources usage."""
        if self._mem_check_proc is not None:
            logger.warning("Resource tracker started already. Please execute start after executing stop.")
            return

        self._queue = mp.Queue()
        self._mem_check_proc = mp.Process(
            target=_check_resource, args=(self._queue, self._resource_type, self._gpu_ids)
        )
        self._mem_check_proc.start()

    def stop(self):
        """Terminate a process to record resources usage."""
        if self._mem_check_proc is None or not self._mem_check_proc.is_alive():
            return

        self._queue.put(self.output_path)
        self._mem_check_proc.join(10)
        if self._mem_check_proc.exitcode is None:
            self._mem_check_proc.terminate()
        self._mem_check_proc.close()

        self._mem_check_proc = None
        self._queue = None


def _check_resource(queue: mp.Queue, resource_types: Optional[List[str]] = None, gpu_ids: Optional[List[int]] = None):
    if resource_types is None:
        resource_types = []

    trackers: Dict[str, ResourceRecorder] = {}
    for resource_type in resource_types:
        if resource_type == "cpu":
            trackers[resource_type] = CpuUsageRecorder()
        elif resource_type == "gpu":
            if pynvml is None:
                logger.warning("GPU can't be found. Tracking GPU usage is skipped.")
                continue
            trackers[resource_type] = GpuUsageRecorder(gpu_ids)
        else:
            raise ValueError(
                "Resource type {} isn't supported now. Current available types are cpu and gpu.".format(resource_type)
            )

    if not trackers:
        logger.warning("There is no resource to record.")
        return

    while True:
        for tracker in trackers.values():
            tracker.record()

        if not queue.empty():
            break

        time.sleep(0.01)

    output_path = Path(queue.get())

    resource_record = {resource_type: tracker.report() for resource_type, tracker in trackers.items()}
    with output_path.open("w") as f:
        yaml.dump(resource_record, f, default_flow_style=False)


class ResourceRecorder(ABC):
    """Base calss for each resource recorder."""

    @abstractmethod
    def record(self):
        """Record a resource usage."""
        raise NotImplementedError

    @abstractmethod
    def report(self):
        """Aggregate all resource usages."""
        raise NotImplementedError


class CpuUsageRecorder(ResourceRecorder):
    """CPU usage recorder class.

    Args:
        target_process Optional[psutil.Process]: Process to track.
    """

    def __init__(self):
        self._record_count: int = 0
        self._max_mem: Union[int, float] = 0
        self._avg_util: Union[int, float] = 0
        self._first_record = True

    def record(self):
        """Record CPU usage."""
        # cpu mem
        memory_info = psutil.virtual_memory()
        cpu_mem = (memory_info.total - memory_info.available) / GIB
        if self._max_mem < cpu_mem:
            self._max_mem = cpu_mem

        # cpu util
        cpu_percent = psutil.cpu_percent()
        if self._first_record:
            self._first_record = False
        else:
            self._avg_util += cpu_percent
            self._record_count += 1

    def report(self) -> Dict[str, str]:
        """Aggregate CPU usage."""
        if self._record_count == 0:
            return {}

        return {
            "max_memory_usage": f"{round(self._max_mem, 2)} GiB",
            "avg_util": f"{round(self._avg_util / self._record_count, 2)} %",
        }


class GpuUsageRecorder(ResourceRecorder):
    """GPU usage recorder class.

    Args:
        gpu_ids Optional[List[int]]: GPU indices to record. If not given, first GPU is recorded.
    """

    def __init__(self, gpu_ids: Optional[List[int]] = None):
        if gpu_ids is None:
            gpu_ids = [0]

        self._record: Dict[int, Dict[str, Union[int, List[int]]]] = {}
        self._gpu_handlers: Dict[int, Any] = {}

        pynvml.nvmlInit()
        gpu_to_track = self._get_gpu_to_track(gpu_ids)
        for gpu_idx in gpu_to_track:
            self._record[gpu_idx] = {"max_mem": 0, "util_record": []}
            self._gpu_handlers[gpu_idx] = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)

    def _get_gpu_to_track(self, gpu_ids: List[int]) -> List[int]:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            avaiable_gpus = [int(idx) for idx in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        else:
            avaiable_gpus = list(range(pynvml.nvmlDeviceGetCount()))
        return [avaiable_gpus[gpu_idx] for gpu_idx in gpu_ids]

    def record(self):
        """Record GPU usage."""
        for gpu_idx, record in self._record.items():
            # gpu util
            gpu_info = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handlers[gpu_idx])
            record["util_record"].append(gpu_info.gpu)

            # gpu mem
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handlers[gpu_idx])
            mem_used = gpu_mem.used / GIB
            if record["max_mem"] < mem_used:
                record["max_mem"] = mem_used

    @no_type_check
    def report(self) -> Dict[str, str]:
        """Aggregate GPU usage."""
        if not list(self._record.values())[0]["util_record"]:  # record isn't called
            return {}

        total_max_mem = 0
        total_avg_util = 0
        gpus_record = self._record.copy()
        for gpu_idx in list(gpus_record.keys()):
            max_mem = gpus_record[gpu_idx]["max_mem"]
            if total_max_mem < max_mem:
                total_max_mem = max_mem

            # Count utilization after it becomes bigger than 20% of max utilization
            max_util = max(gpus_record[gpu_idx]["util_record"])
            for idx, util in enumerate(gpus_record[gpu_idx]["util_record"]):
                if util * 5 > max_util:
                    break
            avg_util = mean(gpus_record[gpu_idx]["util_record"][idx:])
            total_avg_util += avg_util

            gpus_record[f"gpu_{gpu_idx}"] = {
                "avg_util": f"{round(avg_util, 2)} %",
                "max_mem": f"{round(max_mem, 2)} GiB",
            }
            del gpus_record[gpu_idx]

        gpus_record["total_avg_util"] = f"{round(total_avg_util / len(gpus_record), 2)} %"
        gpus_record["total_max_mem"] = f"{round(total_max_mem, 2)} GiB"

        return gpus_record

    def __del__(self):
        """Shutdown nvml."""
        pynvml.nvmlShutdown()
