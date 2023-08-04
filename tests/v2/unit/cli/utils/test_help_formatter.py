# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from otx.v2.cli.utils.arg_parser import OTXArgumentParser
from otx.v2.cli.utils.help_formatter import (
    OTXHelpFormatter,
    get_cli_usage_docstring,
    get_verbose_usage,
    get_verbosity_subcommand,
    pre_parse_arguments,
    render_guide,
)
from pytest_mock.plugin import MockerFixture


def test_pre_parse_arguments(mocker: MockerFixture) -> None:
    argv = ["otx", "subcommand", "--arg1", "value1", "-a2", "value2"]
    mocker.patch.object(sys, "argv", argv)
    expected_output: dict = {
        "subcommand": "subcommand",
        "arg1": "value1",
        "a2": "value2",
    }
    assert pre_parse_arguments() == expected_output


def test_get_verbosity_subcommand(mocker: MockerFixture) -> None:
    argv = ["otx", "train", "-h"]
    mocker.patch.object(sys, "argv", argv)
    assert get_verbosity_subcommand() == (0, "train")

    argv = ["otx", "train", "-h", "-v"]
    mocker.patch.object(sys, "argv", argv)
    assert get_verbosity_subcommand() == (1, "train")

    argv = ["otx", "train", "-h", "-vv"]
    mocker.patch.object(sys, "argv", argv)
    assert get_verbosity_subcommand() == (2, "train")

    argv = ["otx", "-h"]
    mocker.patch.object(sys, "argv", argv)
    assert get_verbosity_subcommand() == (2, None)

def test_get_verbose_usage() -> None:
    subcommand = "test111"
    assert f"otx {subcommand} [optional_arguments]" in get_verbose_usage(subcommand=subcommand)

def test_get_cli_usage_docstring() -> None:
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
    subcommand = "train"
    contents = render_guide(subcommand)
    assert len(contents) == 2
    assert contents[0].__class__.__name__ == "Markdown"
    assert "# OpenVINO™ Training Extensions CLI Guide" in contents[0].markup
    assert contents[1].__class__.__name__ == "Panel"
    assert "1. You must first prepare the dataset" in contents[1].renderable.markup
    assert render_guide(None) == []


class TestOTXHelpFormatter:
    @pytest.fixture()
    def mock_parser(self, monkeypatch: MonkeyPatch) -> OTXArgumentParser:
        parser = OTXArgumentParser()
        mock_arguments = {
            "train": {"never_skip"},
        }
        monkeypatch.setattr("otx.v2.cli.utils.help_formatter.REQUIRED_ARGUMENTS", mock_arguments)
        parser.formatter_class.non_skip_list = {"never_skip"}
        parser.formatter_class.subcommand = "train"
        parser.add_argument(
            "-t",
            "--test",
            action="count",
            help="add_usage test.",
        )
        parser.add_argument(
            "--never_skip",
            action="count",
            help="never_skip test.",
        )
        return parser

    def test_verbose_0(self, capfd: CaptureFixture, mock_parser: OTXArgumentParser) -> None:
        assert mock_parser.formatter_class == OTXHelpFormatter
        mock_parser.formatter_class.verbose_level = 0

        with pytest.raises(SystemExit, match="0"):
            mock_parser.parse_args(["-h"])
        out, _ = capfd.readouterr()
        assert "Quick-Start" in out
        assert "--never_skip" not in out
        assert "--test" not in out

    def test_verbose_1(self, capfd: CaptureFixture, mock_parser: OTXArgumentParser) -> None:
        assert mock_parser.formatter_class == OTXHelpFormatter
        mock_parser.formatter_class.verbose_level = 1

        with pytest.raises(SystemExit, match="0"):
            mock_parser.parse_args(["-h"])
        out, _ = capfd.readouterr()
        assert "--never_skip" in out
        assert "Quick-Start" in out
        assert "--test" not in out

    def test_verbose_2(self, capfd: CaptureFixture, mock_parser: OTXArgumentParser) -> None:
        assert mock_parser.formatter_class == OTXHelpFormatter
        mock_parser.formatter_class.verbose_level = 2

        with pytest.raises(SystemExit, match="0"):
            mock_parser.parse_args(["-h"])
        out, _ = capfd.readouterr()
        assert "--test" in out
        assert "--never_skip" in out
        assert "Quick-Start" not in out
