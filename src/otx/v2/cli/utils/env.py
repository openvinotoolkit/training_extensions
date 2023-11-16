"""OTX environment status & diagnosis."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib

from pkg_resources import Requirement
from rich.console import Console
from rich.table import Table

from otx.v2.adapters import ADAPTERS

from .install import get_module_version, get_requirements

REQUIRED_ADAPTERS_PER_TASK = {
    "anomaly": ["openvino", "datumaro", "torch.lightning.anomalib"],
    "classification": [
        "openvino",
        "datumaro",
        "torch.mmengine",
        "torch.mmengine.mmpretrain",
        "torch.mmengine.mmdeploy",
    ],
    "visual_prompt": [
        "openvino",
        "datumaro",
        "torch.lightning",
    ],
    "segmentation": [
        "openvino",
        "datumaro",
        "torch.mmengine",
        "torch.mmengine.mmseg",
        "torch.mmengine.mmdeploy",
    "detection": [
        "openvino",
        "datumaro",
        "torch.mmengine",
        "torch.mmengine.mmdet",
        "torch.mmengine.mmdeploy",
    ],
}


def get_adapters_status() -> dict[str, dict]:
    """Return the available and version information for each adapter.

    Returns:
        dict[str, dict[str, Union[bool, float]]]: the available and version information.
    """
    adapters_status: dict[str, dict] = {}
    for adapter in ADAPTERS:
        name = f"otx.v2.adapters.{adapter}"
        module = importlib.import_module(name)
        adapters_status[name] = {}
        for var in ("AVAILABLE", "VERSION", "DEBUG"):
            if hasattr(module, var):
                adapters_status[name][var] = getattr(module, var)
    return adapters_status


def get_environment_table(task: str | None = None, verbose: bool | None = None) -> str:
    """Get table provides the availability of each tasks.

    Args:
        verbose (bool, optional): Show more detail dependencies. Defaults to False.

    Returns:
        str: String of rich.table.Table.
    """
    table = Table(title="Current Evironment Status of OTX")
    table.add_column("Task", justify="left", style="yellow")
    table.add_column("Required", justify="left", style="cyan")
    table.add_column("Available", justify="center", style="green")

    task_lst = ["api", "base", "openvino"] if verbose else []
    if task is not None and task.lower() in REQUIRED_ADAPTERS_PER_TASK:
        task_lst.append(task)
    else:
        task_lst.extend(list(REQUIRED_ADAPTERS_PER_TASK.keys()))

    requirements_per_task = get_requirements()
    adapters_status = get_adapters_status()
    requirements: list[str] | list[Requirement]
    for task in task_lst:
        task_name = task
        if verbose:
            if task not in requirements_per_task:
                continue
            requirements = requirements_per_task[task]
        else:
            requirements = REQUIRED_ADAPTERS_PER_TASK[task]
        i = 0
        for req in requirements:
            end_section = i == len(requirements) - 1
            _req = str(req) if isinstance(req, Requirement) else req
            if verbose and isinstance(req, Requirement):
                required = _req
                current_version = get_module_version(req.project_name)
            else:
                required = _req.split(".")[-1]
                adapter = adapters_status[f"otx.v2.adapters.{_req}"]
                current_version = adapter["VERSION"] if adapter["AVAILABLE"] else None
            if current_version is not None:
                table.add_row(task_name, required, current_version, end_section=end_section)
            else:
                table.add_row(task_name, required, "X", style="red", end_section=end_section)
            i += 1
            task_name = ""

    console = Console()
    with console.capture() as capture:
        console.print(table, end="")
    return capture.get()


def get_task_status(task: str | None = None) -> dict[str, dict]:
    """Check if the requirement for each task is currently available.

    Args:
        task (Optional[str], optional): Task available in OTX. Defaults to None.

    Returns:
        Dict[str, Dict[str, Optional[Union[bool, str, List]]]]: Information about availability by task.
    """
    adapter_status = get_adapters_status()

    task_status: dict[str, dict[str, list | bool]] = {}
    target_req: dict[str, list[str]] = {}
    if task is not None:
        target_req[task] = REQUIRED_ADAPTERS_PER_TASK[task]
    else:
        target_req = REQUIRED_ADAPTERS_PER_TASK
    for req, adapters in target_req.items():
        available = True
        exception_lst: list = []
        for adapter in adapters:
            name = f"otx.v2.adapters.{adapter}"
            if name in adapter_status:
                if not adapter_status[name]["AVAILABLE"]:
                    available = False
                    exception_lst.append(adapter_status[name].get("DEBUG", None))
                    break
            else:
                available = False
                exception_lst.append(ModuleNotFoundError(f"{name} not in {list(adapter_status.keys())}"))
                break
        task_status[req] = {"AVAILABLE": available, "EXCEPTIONS": exception_lst}
    return task_status


def check_torch_cuda() -> tuple[float | None, bool]:
    """Information about whether or not TORCH is available.

    Returns:
        tuple[Optional[float], bool]: The version of torch and the value of cuda.is_available().
    """
    has_torch = importlib.util.find_spec("torch")  # type: ignore[attr-defined]
    if not has_torch:
        return None, False
    torch_module = importlib.import_module("torch")
    return torch_module.__version__, torch_module.cuda.is_available()
