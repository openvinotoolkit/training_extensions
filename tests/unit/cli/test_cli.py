# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import pytest
from otx.cli import OTXCLI


class TestOTXCLI:
    def test_init(self, mocker) -> None:
        # Test that main function runs with errors -> return 2
        argv = ["otx"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(
            SystemExit, match="2",
        ):
            OTXCLI()

        argv = ["otx", "-h"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(
            SystemExit, match="0",
        ):
            OTXCLI()

    @pytest.fixture()
    def fxt_train_help_command(self, monkeypatch) -> None:
        argv = ["otx", "train", "-h"]
        monkeypatch.setattr("sys.argv", argv)

    def test_train_help_command(self, fxt_train_help_command) -> None:
        # Test that main function runs with help -> return 0
        with pytest.raises(SystemExit, match="0"):
            OTXCLI()

    @pytest.fixture()
    def fxt_train_command(self, monkeypatch, tmpdir) -> list[str]:
        argv = [
            "otx",
            "train",
            "+recipe=classification/otx_mobilenet_v3_large",
            "checkpoint=my_checkpoint",
            f"base.output_dir={tmpdir}",
        ]
        monkeypatch.setattr("sys.argv", argv)
        return argv

    def test_train_command(self, fxt_train_command, mocker, tmpdir) -> None:
        # Test that main function runs with help -> return 0
        mock_engine_train = mocker.patch("otx.core.engine.train.train")
        cli = OTXCLI()

        assert cli.config["subcommand"] == "train"
        assert "overrides" in cli.config["train"]
        assert cli.config["train"]["overrides"] == fxt_train_command[2:]

        cfg = mock_engine_train.call_args.args[0]
        assert cfg["checkpoint"] == "my_checkpoint"
        assert cfg["base"]["output_dir"] == tmpdir
