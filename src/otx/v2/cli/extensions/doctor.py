"""OTX CLI doctor."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import traceback
from typing import List, Optional

from rich.console import Console

from otx.v2 import __version__ as otx_version
from otx.v2.cli.utils.arg_parser import OTXArgumentParser
from otx.v2.cli.utils.env import check_torch_cuda, get_environment_table, get_task_status
from otx.v2.cli.utils.install import SUPPORTED_TASKS, get_cuda_version


def add_doctor_parser(parser: OTXArgumentParser) -> OTXArgumentParser:
    """Add subparser for doctor command.

    Args:
        parser (OTXArgumentParser): Main ArgumentParser in CLI.

    Returns:
        OTXArgumentParser: Main parser with subparsers merged.
    """
    sub_parser = prepare_parser()
    parser._subcommands_action.add_subcommand(
        "doctor", sub_parser, help="View diagnostic information about the current environment."
    )

    return parser


def prepare_parser() -> OTXArgumentParser:
    """Parses command line arguments.

    Returns:
        OTXArgumentParser: Sub-parser for doctor command.
    """

    parser = OTXArgumentParser()
    parser.add_argument("task", help=f"Supported tasks are: {SUPPORTED_TASKS}.", default=None, type=str)
    parser.add_argument("--debug", help="Print Traceback of Exception for more details.", action="store_true")

    return parser


def doctor(task: Optional[str] = None, debug: bool = False) -> None:
    """Print diagnostic information about the current environment.

    Args:
        task (Optional[str], optional): Task available in OTX. Defaults to None.
    """
    issue_count = 0
    console = Console()
    green_mark = ":white_heavy_check_mark:"
    red_mark = ":x:"
    warning_mark = ":warning:"

    # Print Adapters available table
    env_table = get_environment_table()
    print(env_table)

    # 1. OTX Version
    console.log(f"{green_mark} OTX version: {otx_version}")
    # 2. Check if torch & CUDA Available
    torch_version, cuda_available = check_torch_cuda()
    if torch_version:
        console.log(f"{green_mark} torch version: {torch_version}")
        if cuda_available:
            cuda_version = get_cuda_version()
            console.log(f"\t-> CUDA {cuda_version} is available!")
        else:
            console.log(f"\t-> {warning_mark} CUDA is not available in torch.")
    else:
        console.log(f"{red_mark} torch version: not available.")
        issue_count += 1

    # 3. Check if each task is available
    task_status = get_task_status(task=task)
    for target, status in task_status.items():
        available = status.get("AVAILABLE", None)
        exception_lst: List = status.get("EXCEPTIONS", [])
        if available:
            console.log(f"{green_mark} {target}: [bold green]Ready![/bold green]")
        else:
            for exception in exception_lst:
                if exception is not None:
                    console.log(f"{red_mark} {target}: [bold red]{warning_mark} {exception}[/bold red]")
                if debug and isinstance(exception, Exception):
                    traceback.print_tb(exception.__traceback__)
            console.log(f"\t - Please try this command: 'otx install {target}' or 'otx install full'\n")
            issue_count += 1
    print()

    if issue_count > 0:
        console.log(f":grey_exclamation: Doctor found [bold red]{issue_count}[/bold red] issues.")
    else:
        console.log("[bold green]- No issue found![/bold green]")


def main() -> None:
    """Entry point for OTX CLI doctor."""

    parser = prepare_parser()
    args = parser.parse_args()
    doctor(task=args.task)


if __name__ == "__main__":
    main()
