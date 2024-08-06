"""Custom Help Formatters for OTX CLI."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Iterable

from jsonargparse import DefaultHelpFormatter
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme
from rich_argparse import RichHelpFormatter

if TYPE_CHECKING:
    import argparse

    from rich.console import Console, RenderableType


BASE_ARGUMENTS = {"config", "print_config", "help", "engine", "model", "model.help", "task"}
ENGINE_ARGUMENTS = {"data_root", "engine.help", "engine.device", "work_dir"}
REQUIRED_ARGUMENTS = {
    "train": {
        "data",
        "checkpoint",
        *BASE_ARGUMENTS,
        *ENGINE_ARGUMENTS,
    },
    "test": {
        "data",
        "checkpoint",
        *BASE_ARGUMENTS,
        *ENGINE_ARGUMENTS,
    },
    "predict": {
        "data",
        "checkpoint",
        "return_predictions",
        *BASE_ARGUMENTS,
        *ENGINE_ARGUMENTS,
    },
    "export": {
        "checkpoint",
        "export_format",
        "export_precision",
        "explain",
        "export_demo_package",
        *BASE_ARGUMENTS,
        *ENGINE_ARGUMENTS,
    },
    "optimize": {
        "checkpoint",
        "export_demo_package",
        *BASE_ARGUMENTS,
        *ENGINE_ARGUMENTS,
    },
    "explain": {
        "data",
        "checkpoint",
        "explain_config",
        "dump",
        *BASE_ARGUMENTS,
        *ENGINE_ARGUMENTS,
    },
    "benchmark": {
        "checkpoint",
        *BASE_ARGUMENTS,
        *ENGINE_ARGUMENTS,
    },
}


def get_verbosity_subcommand() -> dict:
    """Parse command line arguments and returns a dictionary of key-value pairs.

    Returns:
        A dictionary containing the parsed command line arguments.

    Examples:
        >>> import sys
        >>> sys.argv = ['otx', 'train', '-h', '-v']
        >>> get_verbosity_subcommand()
        {'subcommand': 'train', 'help': True, 'verbosity': 1}
    """
    arguments: dict = {"subcommand": None, "help": False, "verbosity": 2}
    if len(sys.argv) >= 2 and sys.argv[1] not in ("--help", "-h"):
        arguments["subcommand"] = sys.argv[1]
    if "--help" in sys.argv or "-h" in sys.argv:
        arguments["help"] = True
        if arguments["subcommand"] in REQUIRED_ARGUMENTS:
            arguments["verbosity"] = 0
            if "-v" in sys.argv or "--verbose" in sys.argv:
                arguments["verbosity"] = 1
            if "-vv" in sys.argv:
                arguments["verbosity"] = 2
    return arguments


INTRO_MARKDOWN = (
    "# OpenVINO™ Training Extensions CLI Guide\n\n"
    "Github Repository: [https://github.com/openvinotoolkit/training_extensions](https://github.com/openvinotoolkit/training_extensions)."
    "\n\n"
    "A better guide is provided by the [documentation](https://openvinotoolkit.github.io/training_extensions/stable/)."
)

VERBOSE_USAGE = (
    "To get more overridable argument information, run the command below.\n"
    "```shell\n"
    "# Verbosity Level 1\n"
    ">>> otx {subcommand} [optional_arguments] -h -v\n"
    "# Verbosity Level 2\n"
    ">>> otx {subcommand} [optional_arguments] -h -vv\n"
    "```"
)

CLI_USAGE_PATTERN = r"CLI Usage:(.*?)(?=\n{2,}|\Z)"


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
    match = re.search(CLI_USAGE_PATTERN, component.__doc__, re.DOTALL)
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
    from otx.engine import Engine

    contents: list[Panel | Markdown] = [Markdown(INTRO_MARKDOWN)]
    target_command = getattr(Engine, subcommand)
    cli_usage = get_cli_usage_docstring(target_command)
    if cli_usage is not None:
        cli_usage += f"\n{VERBOSE_USAGE.format(subcommand=subcommand)}"
        quick_start = Panel(Markdown(cli_usage), border_style="dim", title="Quick-Start", title_align="left")
        contents.append(quick_start)
    return contents


class CustomHelpFormatter(RichHelpFormatter, DefaultHelpFormatter):
    """A custom help formatter for OTX CLI.

    This formatter extends the RichHelpFormatter and DefaultHelpFormatter classes to provide
    a more detailed and customizable help output for OTX CLI.

    Attributes:
        verbosity_level : int
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

    verbosity_dict = get_verbosity_subcommand()
    verbosity_level = verbosity_dict["verbosity"]
    subcommand = verbosity_dict["subcommand"]

    def __init__(
        self,
        prog: str,
        indent_increment: int = 2,
        max_help_position: int = 24,
        width: int | None = None,
        console: Console | None = None,
    ) -> None:
        RichHelpFormatter.group_name_formatter = str

        RichHelpFormatter.__init__(self, prog, indent_increment, max_help_position, width, console=console)
        DefaultHelpFormatter.__init__(self, prog, indent_increment, max_help_position, width)

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
        actions = [] if self.verbosity_level == 0 else actions
        if self.subcommand in REQUIRED_ARGUMENTS and self.verbosity_level == 1:
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
            if self.verbosity_level == 0:
                return
            if self.verbosity_level == 1 and action.dest not in REQUIRED_ARGUMENTS[self.subcommand]:
                return
        super().add_argument(action)

    def format_help(self) -> str:
        """Format the help message for the current command and returns it as a string.

        The help message includes information about the command's arguments and options,
        as well as any additional information provided by the command's help guide.

        Returns:
            str: A string containing the formatted help message.
        """
        with self.console.use_theme(Theme(self.styles)), self.console.capture() as capture:
            section = self._root_section
            rendered_content: RenderableType = section
            if self.subcommand in REQUIRED_ARGUMENTS and self.verbosity_level in (0, 1) and len(section.rich_items) > 1:
                contents = render_guide(self.subcommand)
                for content in contents:
                    self.console.print(content)
            if self.verbosity_level > 0:
                if len(section.rich_items) > 1:
                    rendered_content = Panel(section, border_style="dim", title="Arguments", title_align="left")
                self.console.print(rendered_content, highlight=True, soft_wrap=True)
        help_msg = capture.get()

        if help_msg:
            help_msg = self._long_break_matcher.sub("\n\n", help_msg).rstrip() + "\n"
        return help_msg
