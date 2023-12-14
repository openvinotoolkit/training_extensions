"""Custom Help Formatters for OTX CLI."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Iterable

from jsonargparse import DefaultHelpFormatter
from rich.markdown import Markdown
from rich.panel import Panel
from rich_argparse import RichHelpFormatter

from otx.core.config.engine import EngineConfig

if TYPE_CHECKING:
    import argparse

    from rich.console import RenderableType


BASE_ARGUMENTS = {"config", "print_config", "help", "engine", "model", "model.help"}
ENGINE_ARGUMENTS = {f"engine.{field.name}" for field in EngineConfig.__dataclass_fields__.values()}
REQUIRED_ARGUMENTS = {
    "train": {
        "data", "data.task", "data.config.data_root", "checkpoint",
    }.union(BASE_ARGUMENTS).union(ENGINE_ARGUMENTS),
    "test": {
        "data", "data.task", "data.config.data_root", "checkpoint",
    }.union(BASE_ARGUMENTS).union(ENGINE_ARGUMENTS),
}


def pre_parse_arguments() -> dict:
    """Parse command line arguments and returns a dictionary of key-value pairs.

    Returns:
        A dictionary containing the parsed command line arguments.

    Examples:
        >>> import sys
        >>> sys.argv = ['otx', 'train', '--arg1', 'value1', '-a', 'value2', '-h']
        >>> pre_parse_arguments()
        {'subcommand': 'train', 'arg1': 'value1', 'a': 'value2', 'h': None}
    """
    arguments: dict = {"subcommand": None}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            key = sys.argv[i][2:]
            value = None
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                value = sys.argv[i + 1]
                i += 1
            arguments[key] = value
        elif sys.argv[i].startswith("-"):
            key = sys.argv[i][1:]
            value = None
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                value = sys.argv[i + 1]
                i += 1
            arguments[key] = value
        elif i == 1:
            arguments["subcommand"] = sys.argv[i]
        i += 1
    return arguments


def get_verbosity_subcommand() -> tuple:
    """Return a tuple containing the verbosity level and the subcommand name.

    The verbosity level is determined by the command line arguments passed to the script.
    If the subcommand requires additional arguments, the verbosity level is only set if the
    help option is specified. The verbosity level can be set to 0 (no output), 1 (normal output),
    or 2 (verbose output).

    Returns:
        A tuple containing the verbosity level (int) and the subcommand name (str).
    """
    arguments = pre_parse_arguments()
    verbosity = 2
    if arguments["subcommand"] in REQUIRED_ARGUMENTS and ("h" in arguments or "help" in arguments):
        if "v" in arguments:
            verbosity = 1
        elif "vv" in arguments:
            verbosity = 2
        else:
            verbosity = 0
    return verbosity, arguments["subcommand"]


def get_intro() -> Markdown:
    """Return a Markdown object containing the introduction text for OTX CLI Guide.

    The introduction text includes a brief description of the guide and links to the Github repository and documentation

    Returns:
        A Markdown object containing the introduction text for OTX CLI Guide.
    """
    intro_markdown = (
        "# OpenVINO™ Training Extensions CLI Guide\n\n"
        "Github Repository: [https://github.com/openvinotoolkit/training_extensions](https://github.com/openvinotoolkit/training_extensions)."
        "\n\n"
        "A better guide is provided by the [documentation](https://openvinotoolkit.github.io/training_extensions/stable/)."
    )
    return Markdown(intro_markdown)


def get_verbose_usage(subcommand: str = "train") -> str:
    """Return a string containing verbose usage information for the specified subcommand.

    Args:
        subcommand (str): The name of the subcommand to get verbose usage information for. Defaults to "train".

    Returns:
        str: A string containing verbose usage information for the specified subcommand.
    """
    return (
        "To get more overridable argument information, run the command below.\n"
        "```python\n"
        "# Verbosity Level 1\n"
        f"otx {subcommand} [optional_arguments] -h -v\n"
        "# Verbosity Level 2\n"
        f"otx {subcommand} [optional_arguments] -h -vv\n"
        "```"
    )


def get_cli_usage_docstring(component: object | None) -> str | None:
    r"""Get the cli usage from the docstring.

    Args:
        component (Optional[object]): The component to get the docstring from

    Returns:
        Optional[str]: The quick-start guide as Markdown format.

    Example:
        component.__doc__ = '''
            <Prev Section>

            CLI Usage:
                1. First Step.
                2. Second Step.

            <Next Section>
        '''
        >>> get_cli_usage_docstring(component)
        "1. First Step.\n2. Second Step."
    """
    if component is None or component.__doc__ is None or "CLI Usage" not in component.__doc__:
        return None

    pattern = r"CLI Usage:(.*?)(?=\n{2,}|\Z)"
    match = re.search(pattern, component.__doc__, re.DOTALL)

    if match:
        contents = match.group(1).strip().split("\n")
        return "\n".join([content.strip() for content in contents])
    return None


def render_guide(subcommand: str | None = None) -> list:
    """Render a guide for the specified subcommand.

    Args:
        subcommand (Optional[str]): The subcommand to render the guide for.

    Returns:
        list: A list of contents to be displayed in the guide.
    """
    if subcommand is None or subcommand in ("install"):
        return []
    from otx.core.engine import Engine
    contents: list[Panel | Markdown] = [get_intro()]
    target_command = getattr(Engine, subcommand)
    cli_usage = get_cli_usage_docstring(target_command)
    if cli_usage is not None:
        cli_usage += f"\n{get_verbose_usage(subcommand)}"
        quick_start = Panel(Markdown(cli_usage), border_style="dim", title="Quick-Start", title_align="left")
        contents.append(quick_start)
    return contents


class CustomHelpFormatter(RichHelpFormatter, DefaultHelpFormatter):
    """A custom help formatter for OTX CLI.

    This formatter extends the RichHelpFormatter and DefaultHelpFormatter classes to provide
    a more detailed and customizable help output for OTX CLI.

    Attributes:
        verbose_level : int
            The level of verbosity for the help output.
        subcommand : str | None
            The subcommand to render the guide for.

    Methods:
        add_usage(usage, actions, *args, **kwargs)
            Add usage information to the help output.
        add_argument(action)
            Add an argument to the help output.
        format_help()
            Format the help output.
    """

    verbose_level, subcommand = get_verbosity_subcommand()

    def add_usage(self, usage: str | None, actions: Iterable[argparse.Action], *args, **kwargs) -> None:
        """Add usage information to the formatter.

        Args:
            usage (str | None): A string describing the usage of the program.
            actions (Iterable[argparse.Action]): An list of argparse.Action objects.
            *args (Any): Additional positional arguments to pass to the superclass method.
            **kwargs (Any): Additional keyword arguments to pass to the superclass method.

        Returns:
            None
        """
        if self.subcommand in REQUIRED_ARGUMENTS:
            if self.verbose_level == 0:
                actions = []
            elif self.verbose_level == 1:
                actions = [action for action in actions if action.dest in REQUIRED_ARGUMENTS[self.subcommand]]

        super().add_usage(usage, actions, *args, **kwargs)

    def add_argument(self, action: argparse.Action) -> None:
        """Add an argument to the help formatter.

        If the verbose level is set to 0, the argument is not added.
        If the verbose level is set to 1 and the argument is not in the non-skip list, the argument is not added.

        Args:
            action (argparse.Action): The action to add to the help formatter.
        """
        if self.subcommand in REQUIRED_ARGUMENTS:
            if self.verbose_level == 0:
                return
            if self.verbose_level == 1 and action.dest not in REQUIRED_ARGUMENTS[self.subcommand]:
                return
        super().add_argument(action)

    def format_help(self) -> str:
        """Format the help message for the current command and returns it as a string.

        The help message includes information about the command's arguments and options,
        as well as any additional information provided by the command's help guide.

        Returns:
            str: A string containing the formatted help message.
        """
        with self.console.capture() as capture:
            section = self._root_section
            _reder: RenderableType = section
            if self.subcommand in REQUIRED_ARGUMENTS and self.verbose_level in (0, 1) and len(section.rich_items) > 1:
                contents = render_guide(self.subcommand)
                for content in contents:
                    self.console.print(content)
            if self.verbose_level > 0:
                if len(section.rich_items) > 1:
                    _reder = Panel(section, border_style="dim", title="Arguments", title_align="left")
                self.console.print(_reder, highlight=False, soft_wrap=True)
        help_msg = capture.get()

        if help_msg:
            help_msg = self._long_break_matcher.sub("\n\n", help_msg).rstrip() + "\n"
        return help_msg
