"""OTX CLI doctor."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from rich.console import Console

from otx.v2.cli.utils.arg_parser import OTXArgumentParser
from otx.v2.cli.utils.env import get_environment_table, print_task_status
from otx.v2.cli.utils.install import SUPPORTED_TASKS


def add_doctor_parser(parser: OTXArgumentParser) -> OTXArgumentParser:
    """_summary_.

    Args:
        parser (ArgumentParser): _description_
    """
    sub_parser = prepare_parser()
    parser._subcommands_action.add_subcommand("doctor", sub_parser, help="Show OTX Current Environments.")

    return parser


def prepare_parser() -> OTXArgumentParser:
    """Parses command line arguments."""

    parser = OTXArgumentParser()
    parser.add_argument("task", help=f"Supported tasks are: {SUPPORTED_TASKS}.", default="full", type=str)

    return parser


def doctor(task: str):
    """_summary_.

    Args:
        task (str): _description_
    """
    env_table = get_environment_table()
    print(env_table)
    print_task_status()


def main() -> None:
    """Entry point for OTX CLI doctor."""

    parser = prepare_parser()
    args = parser.parse_args()
    doctor(task=args.task)


if __name__ == "__main__":
    main()
