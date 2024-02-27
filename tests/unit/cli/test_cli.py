# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import pytest
import yaml
from otx.cli import OTXCLI, main


class TestOTXCLI:
    def test_init(self, mocker) -> None:
        # Test that main function runs with errors -> return 2
        argv = ["otx"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="2"):
            OTXCLI()

        argv = ["otx", "-h"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="0"):
            OTXCLI()

    def test_main(self, mocker) -> None:
        argv = ["otx"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="2"):
            main()

        argv = ["otx", "-h"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="0"):
            main()

    @pytest.fixture()
    def fxt_train_help_command(self, monkeypatch) -> None:
        argv = ["otx", "train", "-h"]
        monkeypatch.setattr("sys.argv", argv)

    def test_train_help_command(self, fxt_train_help_command) -> None:
        # Test that main function runs with help -> return 0
        with pytest.raises(SystemExit, match="0"):
            OTXCLI()

    def test_init_parser(self, mocker) -> None:
        mocker.patch("otx.cli.cli.OTXCLI.__init__", return_value=None)
        cli = OTXCLI()
        parser = cli.init_parser()
        assert parser.__class__.__name__ == "ArgumentParser"
        argument_list = [action.dest for action in parser._actions]
        expected_argument = ["help", "version"]
        assert argument_list == expected_argument

    def test_subcommand_parser(self, mocker) -> None:
        mocker.patch("otx.cli.cli.OTXCLI.__init__", return_value=None)
        cli = OTXCLI()
        parser, _ = cli.engine_subcommand_parser(subcommand="train")
        assert parser.__class__.__name__ == "ArgumentParser"
        argument_list = [action.dest for action in parser._actions]
        expected_argument = [
            "help",
            "verbose",
            "config",
            "print_config",
            "data_root",
            "task",
            "seed",
            "callback_monitor",
        ]
        for args in expected_argument:
            assert args in argument_list

    def test_add_subcommands(self, mocker) -> None:
        mocker.patch("otx.cli.cli.OTXCLI.__init__", return_value=None)
        cli = OTXCLI()
        cli.parser = cli.init_parser()
        cli._subcommand_method_arguments = {}
        cli.add_subcommands()
        assert cli._subcommand_method_arguments.keys() == cli.engine_subcommands().keys()

    @pytest.fixture()
    def fxt_train_command(self, monkeypatch, tmpdir) -> list[str]:
        argv = [
            "otx",
            "train",
            "--config",
            "src/otx/recipe/detection/atss_mobilenetv2.yaml",
            "--data_root",
            "tests/assets/car_tree_bug",
            "--model.num_classes",
            "3",
            "--work_dir",
            str(tmpdir),
        ]
        monkeypatch.setattr("sys.argv", argv)
        return argv

    def test_instantiate_classes(self, fxt_train_command, mocker) -> None:
        mock_run = mocker.patch("otx.cli.OTXCLI.run")
        cli = OTXCLI()
        assert mock_run.call_count == 1
        cli.instantiate_classes()

        from otx.core.model.entity.base import OTXModel

        assert isinstance(cli.model, OTXModel)

        from otx.core.data.module import OTXDataModule

        assert isinstance(cli.datamodule, OTXDataModule)

        from otx.engine import Engine

        assert isinstance(cli.engine, Engine)

        assert cli.datamodule == cli.engine.datamodule
        assert cli.model == cli.engine.model

    @pytest.fixture()
    def fxt_print_config_scheduler_override_command(self, monkeypatch) -> None:
        argv = [
            "otx",
            "train",
            "--config",
            "src/otx/recipe/detection/atss_mobilenetv2.yaml",
            "--data_root",
            "tests/assets/car_tree_bug",
            "--scheduler.monitor",
            "val/test_f1",
            "--print_config",
        ]
        monkeypatch.setattr("sys.argv", argv)

    def test_print_config_scheduler_override_command(self, fxt_print_config_scheduler_override_command, capfd) -> None:
        # Test that main function runs with help -> return 0
        with pytest.raises(SystemExit, match="0"):
            OTXCLI()
        out, _ = capfd.readouterr()
        result_config = yaml.safe_load(out)
        expected_str = """
        scheduler:
        - class_path: otx.algo.schedulers.LinearWarmupScheduler
          init_args:
              num_warmup_steps: 3
              interval: step
        - class_path: lightning.pytorch.cli.ReduceLROnPlateau
          init_args:
              monitor: val/test_f1
              mode: max
              factor: 0.5
              patience: 5
              threshold: 0.0001
              threshold_mode: rel
              cooldown: 0
              min_lr: 0.0
              eps: 1.0e-08
              verbose: false
        """
        expected_config = yaml.safe_load(expected_str)
        assert expected_config["scheduler"] == result_config["scheduler"]
