"""Utils function for experiments"""

import multiprocessing as mp
import pynvml
import psutil
import yaml
import os
from contextlib import ExitStack
from typing import Union
from pathlib import Path


def run_process_to_check_resource(output_dir: Union[str, Path], exit_stack: ExitStack):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    gpu_used = os.environ.get("CUDA_VISIBLE_DEVICES", 0)

    queue = mp.Queue()
    mem_check_p = mp.Process(target=check_resource, args=(queue, gpu_used))
    mem_check_p.start()

    exit_stack.callback(mem_check_proc_callback, mem_check_p, queue, output_dir)


def mem_check_proc_callback(mem_check_p, queue, output_dir):
    queue.put(output_dir)
    mem_check_p.join(10)
    if mem_check_p.exitcode is None:
        mem_check_p.terminate()
    mem_check_p.close()


def check_resource(queue: mp.Queue, gpu_idx: int = 0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    max_gpu_mem = 0
    avg_gpu_util = 0
    max_cpu_mem = 0
    avg_cpu_util = 0
    gib = 1024**3
    target_process = psutil.Process().parent()

    num_counts = 0
    while True:
        # gpu util
        gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        avg_gpu_util += gpu_info.gpu
        num_counts += 1

        # gpu mem
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = gpu_mem.used / gib
        if max_gpu_mem < mem_used:
            max_gpu_mem = mem_used

        # cpu mem
        # cpu_mem = psutil.virtual_memory()[3] / gib
        # cpu_mem = target_process.memory_percent()
        cpu_mem = target_process.memory_info().rss / gib
        if max_cpu_mem < cpu_mem:
            max_cpu_mem = cpu_mem

        # cpu util
        cpu_percent = target_process.cpu_percent()
        avg_cpu_util += cpu_percent

        if not queue.empty():
            break

    pynvml.nvmlShutdown()
    output_path = Path(queue.get())

    with (output_path / "resource_usage.yaml").open("w") as f:
        yaml.dump(
            {
                "max_cpu_mem(GiB)" : round(max_cpu_mem, 2),
                "avg_cpu_util(%)" : round(avg_cpu_util / num_counts, 2),
                "max_gpu_mem(GiB)" : round(max_gpu_mem, 2),
                "avg_gpu_util(%)" : round(avg_gpu_util / num_counts, 2),
            },
            f,
            default_flow_style=False
        )
