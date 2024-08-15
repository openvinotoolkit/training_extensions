# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from otx.cli import OTXCLI, main
from rich.console import Console


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
    def fxt_train_argv(self, tmpdir) -> list[str]:
        return [
            "otx",
            "train",
            "--config",
            "src/otx/recipe/detection/atss_mobilenetv2.yaml",
            "--data_root",
            "tests/assets/car_tree_bug",
            "--model.label_info",
            "3",
            "--work_dir",
            str(tmpdir),
        ]

    @pytest.fixture()
    def fxt_train_command(self, monkeypatch, fxt_train_argv) -> list[str]:
        monkeypatch.setattr("sys.argv", fxt_train_argv)
        return fxt_train_argv

    def test_instantiate_classes(self, fxt_train_command, mocker) -> None:
        mock_run = mocker.patch("otx.cli.OTXCLI.run")
        cli = OTXCLI()
        assert mock_run.call_count == 1
        cli.instantiate_classes()

        from otx.core.model.base import OTXModel

        assert isinstance(cli.model, OTXModel)

        from otx.core.data.module import OTXDataModule

        assert isinstance(cli.datamodule, OTXDataModule)

        from otx.engine import Engine

        assert isinstance(cli.engine, Engine)

        assert cli.datamodule == cli.engine.datamodule
        assert cli.model == cli.engine.model

    @pytest.mark.parametrize("input_size", [512, 1024])
    def test_instantiate_classes_set_input_size(self, input_size, fxt_train_argv, monkeypatch, mocker) -> None:
        mocker.patch("otx.cli.OTXCLI.run")
        fxt_train_argv.extend(["--data.input_size", str(input_size)])
        monkeypatch.setattr("sys.argv", fxt_train_argv)

        cli = OTXCLI()
        cli.instantiate_classes()

        assert cli.model.input_size == (input_size, input_size)

    @pytest.fixture()
    def mock_model_cls(self) -> MagicMock:
        model_cls = MagicMock()
        model_cls.input_size_multiplier = 12345
        return model_cls

    def test_instantiate_classes_set_adaptive_input_size(
        self,
        fxt_train_argv,
        monkeypatch,
        mocker,
        mock_model_cls,
    ) -> None:
        mocker.patch("otx.cli.OTXCLI.run")
        mocker.patch("otx.utils.utils.get_model_cls_from_config", return_value=mock_model_cls)
        fxt_train_argv.extend(["--data.adaptive_input_size", "auto"])
        monkeypatch.setattr("sys.argv", fxt_train_argv)
        mock_data_module = mocker.patch("otx.core.data.module.adapt_input_size_to_dataset", return_value=1024)

        cli = OTXCLI()
        cli.instantiate_classes()

        assert mock_data_module.call_args.args[-1] == 12345

    def test_raise_error_correctly(self, fxt_train_command, mocker) -> None:
        mock_engine = mocker.patch("otx.cli.OTXCLI.instantiate_engine")
        mock_engine.return_value.train.side_effect = RuntimeError("my_error")

        with pytest.raises(RuntimeError) as exc_info:
            OTXCLI()

        exc_info.match("my_error")

    @pytest.fixture()
    def fxt_print_config_scheduler_override_command(self, monkeypatch) -> None:
        argv = [
            "otx",
            "train",
            "--config",
            "src/otx/recipe/detection/atss_mobilenetv2.yaml",
            "--data_root",
            "tests/assets/car_tree_bug",
            "--model.scheduler.monitor",
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
          class_path: otx.core.schedulers.LinearWarmupSchedulerCallable
          init_args:
            num_warmup_steps: 3
            monitor: val/test_f1
            warmup_interval: step
            main_scheduler_callable:
              class_path: lightning.pytorch.cli.ReduceLROnPlateau
              init_args:
                monitor: val/map_50
                mode: max
                factor: 0.1
                patience: 4
                threshold: 0.0001
                threshold_mode: rel
                cooldown: 0
                min_lr: 0.0
                eps: 1.0e-08
                verbose: deprecated
        """
        expected_config = yaml.safe_load(expected_str)
        assert expected_config["scheduler"] == result_config["model"]["init_args"]["scheduler"]

    @pytest.fixture()
    def fxt_metric_override_command(self, monkeypatch) -> None:
        argv = [
            "otx",
            "train",
            "--config",
            "src/otx/recipe/detection/atss_mobilenetv2.yaml",
            "--data_root",
            "tests/assets/car_tree_bug",
            "--metric",
            "otx.core.metrics.fmeasure.FMeasureCallable",
            "--print_config",
        ]
        monkeypatch.setattr("sys.argv", argv)

    def test_print_metric_override_command(self, fxt_metric_override_command, capfd) -> None:
        # Test that main function runs with help -> return 0
        with pytest.raises(SystemExit, match="0"):
            OTXCLI()
        out, _ = capfd.readouterr()
        result_config = yaml.safe_load(out)
        assert result_config["metric"] == "otx.core.metrics.fmeasure._f_measure_callable"

    def test_print_results(self, mocker, capfd):
        mocker.patch("otx.cli.cli.OTXCLI.__init__", return_value=None)
        cli = OTXCLI()
        cli.console = Console()
        cli.engine = mocker.MagicMock()
        cli.engine.work_dir.return_value = "work_dir"

        cli.subcommand = "train"
        output = {"loss": torch.tensor(0.1), "metric": torch.tensor(0.9)}
        cli._print_results(output)
        out, _ = capfd.readouterr()
        assert "Train metric" in out
        assert "Value" in out
        assert "loss" in out
        assert "metric" in out
        assert "Work Directory:" in out
