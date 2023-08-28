"""OTX environment status & diagnosis."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib

from rich.console import Console
from rich.table import Table

from otx.v2 import __version__ as otx_version
from otx.v2.adapters import ADAPTERS


def get_adapters_status():
    adapters_status = {}
    for adapter in ADAPTERS:
        name = f"otx.v2.adapters.{adapter}"
        module = importlib.import_module(name)
        adapters_status[name] = {"available": module.AVAILABLE, "version": module.VERSION}
    return adapters_status


def get_environment_table():
    table = Table(title="Current Evironment Status of OTX")
    table.add_column("Framework", justify="left", style="yellow")
    table.add_column("Version", justify="left", style="cyan")
    table.add_column("Status", justify="left", style="green")

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


def print_task_status():
    console = Console()
    task_status = {
        "anomaly": ["datumaro", "torch.anomalib"],
        "classification": ["datumaro", "torch.mmengine.mmpretrain"],
        "detection": ["datumaro", "torch.mmengine.mmdetection"],
    }
    adapter_status = get_adapters_status()
    console.log(f":white_heavy_check_mark: OTX Version: {otx_version}")

    for task, adapters in task_status.items():
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
        if available:
            console.log(f":white_heavy_check_mark: {task}: Ready!")
        else:
            console.log(f":x: {task}: :warning:")
            console.log(f"\t - Try command: 'otx install {task}' or 'otx install full'\n")
    print()


if __name__ == "__main__":
    table = get_environment_table()
    print(table)
