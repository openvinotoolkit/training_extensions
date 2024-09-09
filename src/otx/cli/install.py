# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX CLI Installation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser
from pkg_resources import Requirement
from rich.console import Console
from rich.logging import RichHandler

from otx.cli.utils.installation import (
    get_mmcv_install_args,
    get_requirements,
    get_torch_install_args,
    mim_installation,
    parse_requirements,
    patch_mmaction2,
)

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands

logger = logging.getLogger("pip")
logger.setLevel(logging.WARNING)  # setLevel: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
console = Console()
handler = RichHandler(
    console=console,
    show_level=False,
    show_path=False,
)
logger.addHandler(handler)


def add_install_parser(subcommands_action: _ActionSubCommands) -> None:
    """Add subparser for install command.

    Args:
        subcommands_action (_ActionSubCommands): Sub-Command in CLI.

    Returns:
        None
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--option",
        help="Install optional-dependencies. The 'full' option will install all dependencies.",
        default="base",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Set Logger level to INFO",
        action="store_true",
    )
    parser.add_argument(
        "--do-not-install-torch",
        help="Do not install PyTorch. Choose this option if you already installed PyTorch.",
        action="store_true",
    )
    parser.add_argument(
        "--user",
        help="Install packages in the user site directory, e.g., `pip install --user ...`",
        action="store_true",
    )

    subcommands_action.add_subcommand("install", parser, help="Install OTX requirements.")


def otx_install(
    option: str | None = None,
    verbose: bool = False,
    do_not_install_torch: bool = False,
    user: bool = False,
) -> int:
    """Install OTX requirements.

    Args:
        option (str): Optional-dependency to install requirements for.
        verbose (bool): Set pip logger level to INFO
        do_not_install_torch (bool): If true, skip PyTorch installation.
        user (bool): If true, install packages in the user site directory,
            e.g., `pip install --user ...`

    Raises:
        ValueError: When the task is not supported.

    Returns:
        int: Status code of the pip install command.
    """
    from pip._internal.commands import create_command

    requirements_dict = get_requirements("otx")
    # Add base and openvino requirements.
    requirements = requirements_dict["base"]
    requirements_dict.pop("xpu", None)
    if option == "full":
        for extra in requirements_dict:
            requirements.extend(requirements_dict[extra])
    elif option in requirements_dict:
        requirements.extend(requirements_dict[option])
    elif option is not None:
        requirements.append(Requirement.parse(option))

    # Parse requirements into torch, mmcv and other requirements.
    # This is done to parse the correct version of torch (cpu/cuda) and mmcv (mmcv/mmcv-full).
    torch_requirement, mmcv_requirements, other_requirements = parse_requirements(requirements)

    install_args: list[str] = ["--user"] if user else []

    # Combine torch and other requirements.
    install_args = (
        # Get install args for torch to install it from a specific index-url
        other_requirements + get_torch_install_args(torch_requirement)
        if not do_not_install_torch
        else other_requirements
    )

    # Parse mmX requirements if the task requires mmX packages.
    mmcv_install_args = []
    if mmcv_requirements:
        mmcv_install_args = get_mmcv_install_args(torch_requirement, mmcv_requirements)
        install_args += ["openmim"]

    # Install requirements.
    with console.status("[bold green]Working on installation...\n") as status:
        if verbose:
            logger.setLevel(logging.INFO)
            status.stop()
        console.log(f"Installation list: [yellow]{install_args}[/yellow]")
        status_code = create_command("install").main(install_args)
        if status_code == 0:
            console.log(f"Installation Complete: {install_args}")
        else:
            msg = "Cannot complete installation"
            raise RuntimeError(msg)

        # https://github.com/Madoshakalaka/pipenv-setup/issues/101
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

        # Install mmX requirements if the task requires mmX packages using mim.
        if mmcv_install_args and status_code == 0:
            if user:
                mmcv_install_args.append("--user")
            console.log(f"Installation list: [yellow]{mmcv_install_args}[/yellow]")
            status_code = mim_installation(mmcv_install_args)
            if status_code == 0:
                console.log(f"MMLab Installation Complete: {mmcv_install_args}")
            else:
                msg = "Cannot complete installation"
                raise RuntimeError(msg)

    # Patch MMAction2 with src/otx/cli/patches/mmaction2.patch
    patch_mmaction2()

    if status_code == 0:
        console.print("OTX Installation [bold green]Complete.[/bold green]")

    return status_code
