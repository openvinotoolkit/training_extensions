"""OTX environment status & diagnosis."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
from typing import Dict, Optional, Tuple, Union

from rich.console import Console
from rich.table import Table

from otx.v2.adapters import ADAPTERS

REQUIREMENT_PER_TASK = {
    "anomaly": ["datumaro", "torch.anomalib"],
    "classification": ["datumaro", "torch.mmengine.mmpretrain"],
}


def get_adapters_status() -> Dict[str, Dict[str, Union[bool, float]]]:
    """Returns the available and version information for each adapter.

    Returns:
        dict[str, dict[str, Union[bool, float]]]: the available and version information.
    """
    adapters_status = {}
    for adapter in ADAPTERS:
        name = f"otx.v2.adapters.{adapter}"
        module = importlib.import_module(name)
        adapters_status[name] = {"available": module.AVAILABLE, "version": module.VERSION}
    return adapters_status


def get_environment_table() -> str:
    """A table provides the availability of each adapter.

    Returns:
        str: String of rich.table.Table.
    """
    table = Table(title="Current Evironment Status of OTX")
    table.add_column("Adapters", justify="left", style="yellow")
    table.add_column("Version", justify="left", style="cyan")
    table.add_column("Available", justify="center", style="green")

    adapter = get_adapters_status()
    for name, value in adapter.items():
        if value["available"]:
            table.add_row(name, value["version"], "O")
        else:
            table.add_row(name, value["version"], "X", style="red")

    console = Console()
    with console.capture() as capture:
        console.print(table, end="")
    return capture.get()


def get_task_status(task: Optional[str] = None) -> Dict[str, bool]:
    """Check if the requirement for each task is currently available.

    Args:
        task (Optional[str], optional): Task available in OTX. Defaults to None.

    Returns:
        dict[str, bool]: Information about availability by task.
    """
    adapter_status = get_adapters_status()

    task_status = {}
    target_req = {}
    if task is not None:
        target_req[task] = REQUIREMENT_PER_TASK[task]
    else:
        target_req = REQUIREMENT_PER_TASK
    for req, adapters in target_req.items():
        available = True
        for adapter in adapters:
            name = f"otx.v2.adapters.{adapter}"
            if name in adapter_status:
                if not adapter_status[name]["available"]:
                    available = False
                    break
            else:
                available = False
                break
        task_status[req] = available
    return task_status


def check_torch_cuda() -> Tuple[Optional[float], bool]:
    """Information about whether or not TORCH is available.

    Returns:
        tuple[Optional[float], bool]: The version of torch and the value of cuda.is_available().
    """
    has_torch = importlib.util.find_spec("torch")
    if not has_torch:
        return None, False
    torch_module = importlib.import_module("torch")
    return torch_module.__version__, torch_module.cuda.is_available()
