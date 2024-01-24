"""Tests for Custom Help Formatter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import patch

import pytest
from jsonargparse import ArgumentParser
from otx.cli.utils.help_formatter import (
    CustomHelpFormatter,
    get_cli_usage_docstring,
    get_verbosity_subcommand,
    render_guide,
)


def test_get_verbosity_subcommand() -> None:
    """Test if the verbosity level and subcommand are correctly parsed."""
    argv = ["otx", "train", "-h"]
    with patch.object(sys, "argv", argv):
        result = get_verbosity_subcommand()
        assert result["subcommand"] == "train"
        assert result["verbosity"] == 0

    argv = ["otx", "train", "-h", "-v"]
    with patch.object(sys, "argv", argv):
        result = get_verbosity_subcommand()
        assert result["subcommand"] == "train"
        assert result["verbosity"] == 1

    argv = ["otx", "train", "-h", "-vv"]
    with patch.object(sys, "argv", argv):
        result = get_verbosity_subcommand()
        assert result["subcommand"] == "train"
        assert result["verbosity"] == 2

    argv = ["otx", "-h"]
    with patch.object(sys, "argv", argv):
        result = get_verbosity_subcommand()
        assert result["subcommand"] is None
        assert result["verbosity"] == 2


def test_get_cli_usage_docstring() -> None:
    """Test if the CLI usage docstring is correctly parsed."""
    assert get_cli_usage_docstring(None) is None

    class Component:
        """<Prev Section>.

        CLI Usage:
            1. First Step.
            2. Second Step.

        <Next Section>
        """

    assert get_cli_usage_docstring(Component) == "1. First Step.\n2. Second Step."

    class Component2:
        """<Prev Section>.

        CLI Usage-Test:
            test: test.

        <Next Section>
        """

    assert get_cli_usage_docstring(Component2) is None


def test_render_guide() -> None:
    """Test if the guide is correctly rendered."""
    subcommand = "train"
    contents = render_guide(subcommand)
    assert len(contents) == 2
    assert contents[0].__class__.__name__ == "Markdown"
    assert "# OpenVINO™ Training Extensions CLI Guide" in contents[0].markup
    assert contents[1].__class__.__name__ == "Panel"
    assert "otx train" in contents[1].renderable.markup
    assert render_guide(None) == []


class TestCustomHelpFormatter:
    """Test Custom Help Formatter."""

    @pytest.fixture()
    def fxt_parser(self) -> ArgumentParser:
        """Mock ArgumentParser."""
        parser = ArgumentParser(env_prefix="otx", formatter_class=CustomHelpFormatter)
        parser.formatter_class.subcommand = "train"
        parser.add_argument(
            "-t",
            "--test",
            action="count",
            help="add_usage test.",
        )
        parser.add_argument(
            "--model",
            action="count",
            help="never_skip test.",
        )
        return parser

    def test_verbose_0(self, capfd: "pytest.CaptureFixture", fxt_parser: ArgumentParser) -> None:
        """Test verbose level 0."""
        argv = ["otx", "train", "-h"]
        assert fxt_parser.formatter_class == CustomHelpFormatter
        fxt_parser.formatter_class.verbosity_level = 0
        with pytest.raises(SystemExit, match="0"):
            fxt_parser.parse_args(argv)
        out, _ = capfd.readouterr()
        assert "Quick-Start" in out
        assert "Arguments" not in out

    def test_verbose_1(self, capfd: "pytest.CaptureFixture", fxt_parser: ArgumentParser) -> None:
        """Test verbose level 1."""
        argv = ["otx", "train", "-h", "-v"]
        assert fxt_parser.formatter_class == CustomHelpFormatter
        fxt_parser.formatter_class.verbosity_level = 1
        with pytest.raises(SystemExit, match="0"):
            fxt_parser.parse_args(argv)
        out, _ = capfd.readouterr()
        assert "Quick-Start" in out
        assert "Arguments" in out

    def test_verbose_2(self, capfd: "pytest.CaptureFixture", fxt_parser: ArgumentParser) -> None:
        """Test verbose level 2."""
        argv = ["otx", "train", "-h", "-vv"]
        assert fxt_parser.formatter_class == CustomHelpFormatter
        fxt_parser.formatter_class.verbosity_level = 2
        with pytest.raises(SystemExit, match="0"):
            fxt_parser.parse_args(argv)
        out, _ = capfd.readouterr()
        assert "Quick-Start" not in out
        assert "Arguments" in out
