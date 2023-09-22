"""OTX's HelpFormatter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import argparse
import re
from typing import Iterable, Optional

from jsonargparse import DefaultHelpFormatter
from rich.markdown import Markdown
from rich.panel import Panel
from rich_argparse import RichHelpFormatter

from otx.v2.api.core import Engine

# TODO: Let's think about how to manage it more efficiently.
NONSKIP_LIST = {
    "config",
    "print_config",
    "data.train_data_roots",
    "data.val_data_roots",
    "data.test_data_roots",
    "model.name",
    "checkpoint",
    "data.task",
    "img",
    "work_dir",
}


def get_verbose_usage(subcommand: str = "train") -> str:
    return f"""
To get more overridable argument information, run the command below.\n
```python
# Verbosity Level 1
otx {subcommand} [optional_arguments] -h -v
# Verbosity Level 2
otx {subcommand} [optional_arguments] -h -vv
```
"""


def get_cli_usage_docstring(component: Optional[object]) -> Optional[str]:
    """Gets the cli usage from the docstring.

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


def get_intro() -> Markdown:
    intro_markdown = """

# OpenVINO™ Training Extensions CLI Guide

Github Repository: [https://github.com/openvinotoolkit/training_extensions](https://github.com/openvinotoolkit/training_extensions). \n
A better guide is provided by the [documentation](https://openvinotoolkit.github.io/training_extensions/latest/index.html).

"""
    return Markdown(intro_markdown)


def render_guide(subcommand: Optional[str] = None) -> list:
    if subcommand is None:
        return []
    contents = [get_intro()]
    engine_subcommand = getattr(Engine, subcommand, None)
    cli_usage = get_cli_usage_docstring(engine_subcommand)
    if cli_usage is not None:
        cli_usage += f"\n{get_verbose_usage(subcommand)}"
        quick_start = Panel(Markdown(cli_usage), border_style="dim", title="Quick-Start", title_align="left")
        contents.append(quick_start)
    return contents


class OTXHelpFormatter(RichHelpFormatter, DefaultHelpFormatter):
    verbose_level: int = 2
    non_skip_list = NONSKIP_LIST
    subcommand: Optional[str] = None

    def add_usage(self, usage: Optional[str], actions: Iterable[argparse.Action], *args, **kwargs) -> None:
        if self.verbose_level == 0:
            actions = []
        elif self.verbose_level == 1:
            actions = [action for action in actions if action.dest in self.non_skip_list]

        super().add_usage(usage, actions, *args, **kwargs)

    def add_argument(self, action: argparse.Action) -> None:
        if self.verbose_level == 0:
            return
        if self.verbose_level == 1 and action.dest not in self.non_skip_list:
            return
        super().add_argument(action)

    def format_help(self) -> str:
        with self.console.capture() as capture:
            section = self._root_section
            if self.verbose_level in (0, 1) and len(section.rich_items) > 1:
                contents = render_guide(self.subcommand)
                for content in contents:
                    self.console.print(content)
            if self.verbose_level > 0:
                if len(section.rich_items) > 1:
                    section = Panel(section, border_style="dim", title="Arguments", title_align="left")
                self.console.print(section, highlight=False, soft_wrap=True)
        help_msg = capture.get()

        if help_msg:
            help_msg = self._long_break_matcher.sub("\n\n", help_msg).rstrip() + "\n"
        return help_msg
