# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from otx.cli import OTXCLI


class TestOTXCLI:
    def test_init(self, mocker) -> None:
        # Test that main function runs with errors -> return 2
        argv = ["otx"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="2"):
            OTXCLI()

        argv = ["otx", "-h"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="0"):
            OTXCLI()

    def test_subcommand(self, mocker) -> None:
        # Test that main function runs with help -> return 0
        argv = ["otx", "train", "-h"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="0"):
            OTXCLI()

        argv = ["otx", "train", "+recipe=test", "model=model1"]
        mocker.patch.object(sys, "argv", argv)
        mock_otx_train = mocker.patch("otx.cli.train.otx_train")
        cli = OTXCLI()
        assert cli.config["subcommand"] == "train"
        assert "overrides" in cli.config["train"]
        assert cli.config["train"]["overrides"] == ["+recipe=test", "model=model1"]
        mock_otx_train.assert_called_once_with(overrides=["+recipe=test", "model=model1"])
