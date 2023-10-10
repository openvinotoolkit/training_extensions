"""OTX CLI Installation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pip._internal.commands import create_command

from otx.v2.cli.utils.arg_parser import OTXArgumentParser
from otx.v2.cli.utils.install import (
    SUPPORTED_TASKS,
    get_mmcv_install_args,
    get_requirements,
    get_torch_install_args,
    mim_installation,
    parse_requirements,
)

if TYPE_CHECKING:  # pragma: no cover
    from jsonargparse._actions import _ActionSubCommands


def add_install_parser(subcommands_action: _ActionSubCommands) -> None:
    """Add subparser for install command.

    Args:
        subcommands_action (_ActionSubCommands): Sub-Command in CLI.

    Returns:
        None
    """
    sub_parser = prepare_parser()
    subcommands_action.add_subcommand("install", sub_parser, help="Install OTX requirements.")


def prepare_parser() -> OTXArgumentParser:
    """Return an instance of OTXArgumentParser with the required arguments for the install command.

    :return: An instance of OTXArgumentParser.
    """
    parser = OTXArgumentParser()
    parser.add_argument("task", help=f"Supported tasks are: {SUPPORTED_TASKS}.", default="full", type=str)

    return parser


def install(task: str) -> int:
    """Install OTX requirements.

    Args:
        task (str): Task to install requirements for.

    Raises:
        ValueError: When the task is not supported.

    Returns:
        int: Status code of the pip install command.
    """
    requirements_dict = get_requirements("otx")
    # Add base and openvino requirements.
    requirements = requirements_dict["base"]
    if task == "full":
        requirements.extend(requirements_dict["openvino"])
        for extra in SUPPORTED_TASKS:
            requirements.extend(requirements_dict[extra])
    elif task in SUPPORTED_TASKS:
        requirements.extend(requirements_dict["openvino"])
        requirements.extend(requirements_dict[task])
    elif task in requirements_dict:
        requirements.extend(requirements_dict[task])
    else:
        msg = f"Invalid task type: {task}. Supported tasks: {SUPPORTED_TASKS}"
        raise ValueError(msg)

    # Parse requirements into torch, mmcv and other requirements.
    # This is done to parse the correct version of torch (cpu/cuda) and mmcv (mmcv/mmcv-full).
    # TODO: Check pre-installed torch with Intel-Device (eg. torch with IPEX).
    torch_requirement, mmcv_requirements, other_requirements = parse_requirements(requirements)

    # Get install args for torch to install it from a specific index-url
    install_args: list[str] = []
    torch_install_args = get_torch_install_args(torch_requirement)

    # Combine torch and other requirements.
    install_args = other_requirements + torch_install_args

    # Parse mmX requirements if the task requires mmX packages.
    mmcv_install_args = []
    if mmcv_requirements:
        mmcv_install_args = get_mmcv_install_args(torch_requirement, mmcv_requirements)
        install_args += ["openmim"]

    # Install requirements.
    status_code = create_command("install").main(install_args)

    # FIXME: Issue with setuptools - https://github.com/Madoshakalaka/pipenv-setup/issues/101
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    # Install mmX requirements if the task requires mmX packages using mim.
    if mmcv_install_args:
        mim_installation(mmcv_install_args)

    return status_code


def main() -> None:
    """Entry point for OTX CLI Install."""
    parser = prepare_parser()
    args = parser.parse_args()
    install(task=args.task)


if __name__ == "__main__":
    main()
