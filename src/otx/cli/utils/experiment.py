"""Utils function for experiments"""

import multiprocessing as mp
import pynvml
import psutil
from contextlib import ExitStack
from typing import Union
from pathlib import Path


def run_process_to_check_resource(output_dir: Union[str, Path], exit_stack: ExitStack):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    queue = mp.Queue()
    mem_check_p = mp.Process(target=check_resource, args=(queue,))
    mem_check_p.start()

    exit_stack.callback(mem_check_proc_callback, mem_check_p, queue, output_dir)


def mem_check_proc_callback(mem_check_p, queue, output_dir):
    queue.put(output_dir / "resource.txt")
    mem_check_p.join(10)
    if mem_check_p.exitcode is None:
        mem_check_p.terminate()
    mem_check_p.close()


def check_resource(queue: mp.Queue):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
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
        cpu_mem = psutil.virtual_memory()[3] / gib
        if max_cpu_mem < cpu_mem:
            max_cpu_mem = cpu_mem

        # cpu util
        cpu_percent = psutil.cpu_percent()
        avg_cpu_util += cpu_percent

        if not queue.empty():
            break

    pynvml.nvmlShutdown()
    output_path = queue.get()

    with open(output_path, "w") as f:
        f.write(
            f"max_cpu_mem\t{max_cpu_mem} GiB\n"
            f"avg_cpu_util\t{avg_cpu_util / num_counts} %\n"
            f"max_gpu_mem\t{max_gpu_mem} GiB\n"
            f"avg_gpu_util\t{avg_gpu_util / num_counts} %\n"
        )

