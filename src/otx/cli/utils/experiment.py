"""Utils function for experiments"""

import multiprocessing as mp
import psutil
import yaml
import logging
import os
from abc import ABC, abstractmethod
from typing import Union, Optional, List, Dict, Any
from pathlib import Path

try:
    import pynvml
except ImportError:
    pynvml = None

logger = logging.getLogger(__name__)
GIB = 1024**3


class ResourceTracker:
    """Class to track resources usage.

    Args:
        output_dir (Union[str, Path]): Output directory path where CPU & GPU utilization and max meory usage values
                                    are saved.
        gpu_ids (Optional[str]): GPU indices to record.
    """
    def __init__(self, output_dir: Union[str, Path], gpu_ids: Optional[str] = None):
        self._output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
        self._gpu_ids = gpu_ids
        self._mem_check_proc: Union[mp.Process, None] = None
        self._queue: Union[mp.Queue, None] = None

    def start(self):
        """Run a process which tracks resources usage"""
        self._queue = mp.Queue()
        self._mem_check_proc = mp.Process(target=_check_resource, args=(self._queue, ["cpu", "gpu"], self._gpu_ids))
        self._mem_check_proc.start()

    def stop(self):
        """Terminate a process to record resources usage."""
        if self._mem_check_proc is None or not self._mem_check_proc.is_alive():
            return

        self._queue.put(self._output_dir)
        self._mem_check_proc.join(10)
        if self._mem_check_proc.exitcode is None:
            self._mem_check_proc.terminate()
        self._mem_check_proc.close()

        self._mem_check_proc = None
        self._queue = None


def _check_resource(
    queue: mp.Queue,
    resource_types: Optional[Union[str, List[str]]] = None,
    gpu_ids: Optional[List[int]] = None
):
    if resource_types is None:
        resource_types = []

    trackers: Dict[str, ResourceRecorder] = {}
    for resource_type in resource_types:
        if resource_type == "cpu":
            trackers[resource_type] = CpuUsageRecorder(psutil.Process().parent())
        elif resource_type == "gpu":
            if pynvml is None:
                logger.warning("GPU can't be found. Tracking GPU usage is skipped.")
                continue
            trackers[resource_type] = GpuUsageRecorder(gpu_ids)
        else:
            logger.warning(
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

    output_path = Path(queue.get())

    resource_record = {resource_type : tracker.report() for resource_type, tracker in trackers.items()}
    with (output_path / "resource_usage.yaml").open("w") as f:
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
    def __init__(self, target_process: Optional[psutil.Process] = None):
        self._target_process = psutil.Process() if target_process is None else target_process
        self._record_count: int = 0
        self._max_mem: Union[int, float] = 0
        self._avg_util: Union[int, float] = 0

    def record(self):
        """Record CPU usage."""
        # cpu mem
        cpu_mem = self._target_process.memory_info().rss / GIB
        if self._max_mem < cpu_mem:
            self._max_mem = cpu_mem

        # cpu util
        cpu_percent = self._target_process.cpu_percent()
        if self._record_count != 0:  # a value at the first time is meaningless
            self._avg_util += cpu_percent

        self._record_count += 1

    def report(self) -> Dict[str, str]:
        """Aggregate CPU usage."""
        if self._record_count == 0:
            return {}

        return {
            "max_memory_usage" : f"{round(self._max_mem, 2)} GiB",
            "avg_util" : f"{round(self._avg_util / self._record_count, 2)} %"
        }


class GpuUsageRecorder(ResourceRecorder):
    """GPU usage recorder class.

    Args:
        gpu_ids Optional[List[int]]: GPU indices to record. If not given, first GPU is recorded.
    """
    def __init__(self, gpu_ids: Optional[List[int]] = None):
        if gpu_ids is None:
            gpu_ids = [0]

        self._record_count: int = 0
        self._record: Dict[str, Union[int, float]] = {}
        self._gpu_handlers: Dict[int, Any] = {}

        pynvml.nvmlInit()
        gpu_to_track = self._get_gpu_to_track(gpu_ids)
        for gpu_idx in gpu_to_track:
            self._record[gpu_idx] = {"max_mem" : 0, "avg_util" : 0} 
            self._gpu_handlers[gpu_idx] = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)

    def _get_gpu_to_track(self, gpu_ids: List[int]) -> List[int]:
        avaiable_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
        if avaiable_gpus is None:
            avaiable_gpus = list(range(pynvml.nvmlDeviceGetCount()))
        else:
            avaiable_gpus = [int(idx) for idx in avaiable_gpus.split(',')]
        return [avaiable_gpus[gpu_idx] for gpu_idx in gpu_ids]

    def record(self):
        """Record GPU usage."""
        for gpu_idx, record in self._record.items():
            # gpu util
            gpu_info = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handlers[gpu_idx])
            record["avg_util"] += gpu_info.gpu

            # gpu mem
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handlers[gpu_idx])
            mem_used = gpu_mem.used / GIB
            if record["max_mem"] < mem_used:
                record["max_mem"] = mem_used

        self._record_count += 1

    def report(self) -> Dict[str, str]:
        """Aggregate GPU usage."""
        if self._record_count == 0:
            return {}

        total_max_mem = 0
        total_avg_util = 0
        gpus_record = self._record.copy()
        for gpu_idx in list(gpus_record.keys()):
            max_mem = gpus_record[gpu_idx]['max_mem']
            if total_max_mem < max_mem:
                total_max_mem = max_mem
            avg_util = gpus_record[gpu_idx]['avg_util'] / self._record_count
            total_avg_util = avg_util

            gpus_record[gpu_idx]["avg_util"] = f"{round(avg_util, 2)} %"
            gpus_record[gpu_idx]["max_mem"] = f"{round(max_mem, 2)} GiB"
            gpus_record[f"gpu_{gpu_idx}"] = gpus_record[gpu_idx]
            del gpus_record[gpu_idx]

        gpus_record["total_avg_util"] = f"{round(total_avg_util / len(gpus_record), 2)} %"
        gpus_record["total_max_mem"] = f"{round(total_max_mem, 2)} GiB"

        return gpus_record

    def __del__(self):
        """Shutdown nvml."""
        pynvml.nvmlShutdown()
