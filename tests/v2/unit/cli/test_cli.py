# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

import pytest
from _pytest.monkeypatch import MonkeyPatch
from jsonargparse import Namespace
from otx.v2.api.entities.task_type import TaskType
from otx.v2.cli.cli import OTXCLIv2, main
from pytest_mock.plugin import MockerFixture


class MockDataset:
    def __init__(self, *args, **kwargs) -> None:
        _, _ = args, kwargs
        self.num_classes = 10

    def train_dataloader(self, *args, **kwargs) -> None:
        _, _ = args, kwargs
    def val_dataloader(self, *args, **kwargs) -> None:
        _, _ = args, kwargs
    def test_dataloader(self, *args, **kwargs) -> None:
        _, _ = args, kwargs

class MockWorkspace:
    def __init__(self) -> None:
        self.work_dir = "test/path/work_dir"
    def add_config(self, *args, **kwargs) -> None:
        pass
    def dump_config(self, *args, **kwargs) -> None:
        pass
    def update_latest(self, *args, **kwargs) -> None:
        pass


class TestCLIv2:
    @pytest.fixture(autouse=True)
    def setup(self, mocker: MockerFixture) -> None:
        # To avoid unnecessary output from rich-Console
        mocker.patch("otx.v2.cli.cli.Console", return_value=mocker.MagicMock())
        mocker.patch("rich.console.Console.print", return_value=None)
        mocker.patch("otx.v2.cli.cli.CLI_EXTENSIONS", return_value={})

    @pytest.fixture()
    def mock_auto_runner(self, mocker: MockerFixture) -> tuple:
        class MockEngine:
            def __init__(self, *args, **kwargs) -> None:
                _, _ = args, kwargs
                self.count = 0
                self.config = {"model": {"name": "test_model"}}
            def train(self, *args, **kwargs) -> dict:
                _, _ = args, kwargs
                self.count += 1
                return {"checkpoint": "test/path/ckpt", "model": mocker.MagicMock()}
            def val(self, *args, **kwargs) -> dict:
                _, _ = args, kwargs
                self.count += 1
                return {"score": 100}
            def test(self, *args, **kwargs) -> dict:
                _, _ = args, kwargs
                self.count += 1
                return {"score": 100}
            def predict(self, *args, **kwargs) -> dict:
                _, _ = args, kwargs
                self.count += 1
                return {"pred_score": 100}
            def export(self, *args, **kwargs) -> dict:
                _, _ = args, kwargs
                self.count += 1
                return {"openvino": "test/openvino/xml"}

        class MockAutoRunner:
            def __init__(self, use_engine: bool=True) -> None:
                self.engine = MockEngine() if use_engine else None
                self.config_path = "test/path/default_config.yaml"
                self.config_list = {"test_model": "test/path/test_model.yaml"}
                self.task = TaskType.CLASSIFICATION
                self.config = {"model": {"name": "test_model"}}

            def build_framework_engine(self) -> None:
                pass

            def get_model(self, model: Union[str, dict], *args, **kwargs) -> Optional[mocker.MagicMock]:
                _, _ = args, kwargs
                if model == "test_model" or isinstance(model, dict):
                    return mocker.MagicMock()
                return None
        return MockAutoRunner, MockEngine

    def test_init(self, mocker: MockerFixture) -> None:
        # Test that main function runs with errors -> return 2
        argv = ["otx"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="2"):
            OTXCLIv2()

        argv = ["otx", "-h"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="0"):
            OTXCLIv2()

    def test_main(self, mocker: MockerFixture) -> None:
        argv = ["otx"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="2"):
            main()

        argv = ["otx", "-h"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="0"):
            main()

    def test_subcommand_init(self, mocker: MockerFixture) -> None:
        # Test that main function runs with help -> return 0
        argv = ["otx", "train", "-h"]
        with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="0"):
            OTXCLIv2()

        # with mocker.patch.object(sys, "argv", argv) and pytest.raises(SystemExit, match="0"):

        mocker.patch("otx.v2.cli.cli.OTXCLIv2.parse_arguments", return_value={"subcommand": "train"})
        mocker.patch("otx.v2.cli.cli.OTXCLIv2.get_model_class", return_value=(mocker.MagicMock(), []))
        mocker.patch("otx.v2.cli.cli.OTXArgumentParser.add_core_class_args", return_value=None)
        mocker.patch("otx.v2.cli.cli.OTXArgumentParser.set_defaults", return_value=None)
        cli = OTXCLIv2()
        assert cli.subcommand == "train"

        class MockAutoRunner:
            def __init__(self) -> None:
                self.framework_engine = mocker.MagicMock()
                self.dataset_class = mocker.MagicMock()

            def build_framework_engine(self) -> None:
                pass

        mocker.patch("otx.v2.cli.cli.OTXCLIv2.get_auto_runner", return_value=MockAutoRunner())
        mocker.patch("otx.v2.cli.cli.OTXCLIv2.setup_parser")
        cli = OTXCLIv2()
        assert cli.subcommand == "train"

    def test_init_parser(self, mocker: MockerFixture) -> None:
        mocker.patch("otx.v2.cli.cli.OTXCLIv2.__init__", return_value=None)
        cli = OTXCLIv2()
        parser = cli.init_parser()
        assert parser.__class__.__name__ == "OTXArgumentParser"
        argument_list = [action.dest for action in parser._actions]
        expected_argument = ["help", "version", "verbose", "config", "print_config", "work_dir", "framework"]
        assert argument_list == expected_argument

    def test_set_extension_subcommands_parser(self, mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
        def mock_add_extension_parser(*args, **kwargs) -> tuple:
            return args, kwargs

        def mock_extension_main(*args, **kwargs) -> tuple:
            return args, kwargs

        mock_extensions = {
            "test1": {
                "add_parser": mock_add_extension_parser,
                "main": mock_extension_main,
            },
        }

        monkeypatch.setattr("otx.v2.cli.cli.CLI_EXTENSIONS", mock_extensions)
        mocker.patch("otx.v2.cli.cli.OTXCLIv2.__init__", return_value=None)
        cli = OTXCLIv2()
        cli.parser_subcommands = mocker.MagicMock()
        cli._set_extension_subcommands_parser()

        mock_extensions = {
            "test1": {
                "add_parser": None,
            },
        }
        monkeypatch.setattr("otx.v2.cli.cli.CLI_EXTENSIONS", mock_extensions)
        with pytest.raises(NotImplementedError):
            cli._set_extension_subcommands_parser()

    def test_get_auto_runner(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.cli.cli.OTXCLIv2.__init__", return_value=None)
        cli = OTXCLIv2()
        cli.pre_args = {"subcommand": "train", "checkpoint": "test/path/ckpt"}
        with pytest.raises(FileNotFoundError, match="Double-check your checkpoint file."):
            cli.get_auto_runner()

        tmp_dir = tmp_dir_path / "test_checkpoint_check"
        tmp_dir.mkdir()
        (tmp_dir / "test_ckpt.pth").touch()
        cli.pre_args = {"subcommand": "train", "checkpoint": str(tmp_dir / "test_ckpt.pth")}
        with pytest.raises(FileNotFoundError, match="Please include --config."):
            cli.get_auto_runner()

        mocker.patch("otx.v2.cli.cli.Path.exists", return_value=True)
        cli.get_auto_runner()
        assert "config" in cli.pre_args
        assert cli.pre_args["config"] == str(tmp_dir / "configs.yaml")

        cli.pre_args = {"subcommand": "train", "checkpoint": None}
        cli.auto_runner_class = mocker.MagicMock()
        cli.error = None
        cli.get_auto_runner()
        assert cli.error is None
        cli.auto_runner_class.side_effect = ValueError()
        cli.get_auto_runner()
        assert cli.error.__class__.__name__ == "ValueError"

    def test_get_model_class(self, mocker: MockerFixture, mock_auto_runner: tuple) -> None:
        mocker.patch("otx.v2.cli.cli.OTXCLIv2.__init__", return_value=None)

        auto_runner, _ = mock_auto_runner

        cli = OTXCLIv2()
        cli.auto_runner = auto_runner()
        # From model.name
        cli.pre_args = {"model.name": "test_model"}
        model_class, default_configs = cli.get_model_class()
        assert model_class.__name__ == "MagicMock"
        assert len(default_configs) == 2

        # From framework_engine's config model
        cli.pre_args = {}
        model_class, default_configs = cli.get_model_class()
        assert model_class.__name__ == "MagicMock"
        assert len(default_configs) == 2

        # Not supported model case (model is None)
        cli.pre_args = {"model.name": "test_model_2"}
        cli.auto_runner = auto_runner()
        with pytest.raises(ValueError, match="The model was not built"):
            cli.get_model_class()

    def test_parse_arguments(self, mocker: MockerFixture) -> None:
        mocker.patch("otx.v2.cli.cli.OTXCLIv2.__init__", return_value=None)
        mock_object = mocker.patch("otx.v2.cli.cli.OTXArgumentParser.parse_object", return_value=None)
        mock_args = mocker.patch("otx.v2.cli.cli.OTXArgumentParser.parse_args", return_value=None)
        cli = OTXCLIv2()
        parser = cli.init_parser()
        cli.parse_arguments(parser, args={"subcommand": "train"})
        mock_object.assert_called_once()

        cli.parse_arguments(parser, args=None)
        mock_args.assert_called_once()

    def test_instantiate_classes(self, mocker: MockerFixture, mock_auto_runner: tuple) -> None:
        mocker.patch("otx.v2.cli.cli.OTXCLIv2.__init__", return_value=None)
        cli = OTXCLIv2()
        cli.parser = cli.init_parser()
        cli.config = None
        cli.subcommand = "train"
        cli.auto_runner = None
        cli.error = None
        with pytest.raises(ValueError, match="Couldn't run because it couldn't find a suitable task."):
            cli.instantiate_classes("train")

        cli.error = NotImplementedError("test")
        with pytest.raises(NotImplementedError, match="test"):
            cli.instantiate_classes("train")

        auto_runner, _ = mock_auto_runner
        cli.auto_runner = auto_runner()
        config_init = Namespace(model={"name": "test_model"})
        mocker.patch("otx.v2.cli.cli.OTXArgumentParser.instantiate_classes", return_value=config_init)
        with pytest.raises(TypeError, match="There is a problem with data configuration."):
            cli.instantiate_classes("train")

        config_init = Namespace(data={"path": "test/data/path"})
        mocker.patch("otx.v2.cli.cli.OTXArgumentParser.instantiate_classes", return_value=config_init)
        cli.data_class = MockDataset
        with pytest.raises(TypeError, match="There is a problem with model configuration."):
            cli.instantiate_classes("train")

        config_init = Namespace(data={"path": "test/data/path"}, model={"name": "test_model", "head": Namespace(num_classes=5)}, config=["test/path/config.yaml"], work_dir=None)
        mocker.patch("otx.v2.cli.cli.OTXArgumentParser.instantiate_classes", return_value=config_init)
        mock_workspace = mocker.patch("otx.v2.cli.cli.Workspace", return_value=MockWorkspace())
        cli.data_class = MockDataset
        cli.framework_engine = mocker.MagicMock()
        cli.instantiate_classes("train")
        mock_workspace.assert_called_once()
        cli.framework_engine.assert_called_once()

    def test_run_extensions(self, mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
        mocker.patch("otx.v2.cli.cli.OTXCLIv2.__init__", return_value=None)
        cli = OTXCLIv2()
        def mock_add_extension_parser(*args, **kwargs) -> tuple:
            return args, kwargs

        def mock_extension_main(*args, **kwargs) -> tuple:
            return args, kwargs

        mock_extensions = {
            "test1": {
                "add_parser": mock_add_extension_parser,
                "main": mock_extension_main,
            },
        }
        monkeypatch.setattr("otx.v2.cli.cli.CLI_EXTENSIONS", mock_extensions)
        mocker.patch("otx.v2.cli.cli.namespace_to_dict", return_value={})
        cli.config = {"test1": {"test": "test"}}
        cli.run(subcommand="test1")

        mock_extensions["test1"]["main"] = None
        monkeypatch.setattr("otx.v2.cli.cli.CLI_EXTENSIONS", mock_extensions)
        with pytest.raises(NotImplementedError, match="not implemented"):
            cli.run(subcommand="test1")

    @pytest.mark.parametrize("subcommand", ["train", "test", "predict", "export"])
    def test_run_engine(self, mocker: MockerFixture, mock_auto_runner: tuple, subcommand: str) -> None:
        argv = ["otx", subcommand]
        mocker.patch.object(sys, "argv", argv)
        mock_workspace = mocker.patch("otx.v2.cli.cli.Workspace", return_value=MockWorkspace())

        auto_runner, engine = mock_auto_runner
        cli = OTXCLIv2()
        cli.data_class = MockDataset
        cli.framework_engine = engine
        cli.auto_runner = auto_runner()
        cli.run(subcommand=subcommand)
        assert cli.engine.count == 1
        mock_workspace.assert_called_once()

    def test_run_non_valid_subcommand(self, mocker: MockerFixture) -> None:
        argv = ["otx", "train"]
        mocker.patch.object(sys, "argv", argv)
        cli = OTXCLIv2()
        with pytest.raises(NotImplementedError, match="trein is not implemented."):
            cli.run(subcommand="trein")

    def test_version_output(self) -> None:
        result = subprocess.run(args=["otx", "-V"], capture_output=True, text=True, check=False)
        assert result.returncode == 0
        assert "otx" in result.stdout

    def test_help_output(self) -> None:
        result = subprocess.run(args=["otx", "-h"], capture_output=True, text=True, check=False)
        assert result.returncode == 0
        assert "Usage" in result.stdout

    def test_print_config_output(self) -> None:
        result = subprocess.run(args=["otx", "train", "--print_config"], capture_output=True, text=True, check=False)
        assert result.returncode == 0
        assert "work_dir" in result.stdout
        assert "framework" in result.stdout
