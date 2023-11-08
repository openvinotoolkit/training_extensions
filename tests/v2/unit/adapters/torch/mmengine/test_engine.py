# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from pytest_mock.plugin import MockerFixture


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._config = Config({})


class TestMMXEngine:
    def test_init(self, tmp_dir_path: Path) -> None:
        engine = MMXEngine(work_dir=tmp_dir_path)
        assert engine.work_dir == tmp_dir_path
        assert engine.registry.name == "mmengine"
        assert engine.timestamp is not None

    def test_get_value_from_config(self, tmp_dir_path: Path) -> None:
        engine = MMXEngine(work_dir=tmp_dir_path)
        engine.default_config = Config({"max_epochs": 20})
        result = engine._get_value_from_config(
            arg_key="max_epochs",
            positional_args={"max_epochs": 10},
        )

        assert result == 10

        result = engine._get_value_from_config(
            arg_key="max_epochs",
            positional_args={},
        )

        assert result == 20

        engine.default_config = Config({})
        result = engine._get_value_from_config(
            arg_key="max_epochs",
            positional_args={},
        )

        assert result is None

    def test_update_train_config_with_max_epochs(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.mmengine.engine.get_device", return_value="cuda")
        precision = "float32"
        updated_config = Config({"test": "test1"})
        mock_dataloader = mocker.MagicMock()
        engine = MMXEngine(work_dir=tmp_dir_path)
        engine.default_config = Config({"max_epochs": 10, "val_interval": 2, "precision": "float32"})
        engine._update_train_config(
            train_dataloader=mock_dataloader,
            arguments={},
            config=updated_config,
        )

        assert updated_config["train_cfg"]["by_epoch"] is True
        assert updated_config["train_cfg"]["max_epochs"] == engine.default_config["max_epochs"]
        assert updated_config["optim_wrapper"]["type"] == "AmpOptimWrapper"
        assert updated_config["optim_wrapper"]["dtype"] == precision


    def test_update_train_config_with_max_iters(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.mmengine.engine.get_device", return_value="cuda")
        config = {"max_epochs": 10, "val_interval": 2}
        func_args = {}
        func_args["max_iters"] = 100
        func_args["max_epochs"] = 100
        func_args["precision"] = "float16"
        engine = MMXEngine(work_dir=tmp_dir_path)
        updated_config = Config({"test": "test1"})
        mock_dataloader = mocker.MagicMock()
        with pytest.raises(ValueError, match="Only one of `max_epochs` or `max_iters`"):
            engine._update_train_config(
                train_dataloader=mock_dataloader,
                arguments=func_args,
                config=updated_config,
            )
        config["max_epochs"] = None
        func_args["max_epochs"] = None
        engine._update_train_config(
            train_dataloader=mock_dataloader,
            arguments=func_args, config=updated_config,
        )

        assert updated_config["train_cfg"]["by_epoch"] is False
        assert updated_config["train_cfg"]["max_iters"] == func_args["max_iters"]
        assert updated_config["optim_wrapper"]["type"] == "AmpOptimWrapper"
        assert updated_config["optim_wrapper"]["dtype"] == func_args["precision"]


    def test_update_train_config_raises_value_error(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        engine = MMXEngine(work_dir=tmp_dir_path)
        config = {"val_interval": 2}
        config["max_iters"] = 100
        config["max_epochs"] = 10
        engine.default_config = Config(config)

        mock_dataloader = mocker.MagicMock()
        result_config = Config({})
        with pytest.raises(ValueError, match="Only one of `max_epochs` or `max_iters`"):
            engine._update_train_config(
                train_dataloader=mock_dataloader,
                arguments={},
                config=result_config,
            )


    def test_update_train_config_with_train_cfg_in_kwargs(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        engine = MMXEngine(work_dir=tmp_dir_path)
        engine.default_config = Config({"val_interval": 3})
        updated_config = Config({})
        func_args = {}
        mock_dataloader = mocker.MagicMock()
        engine._update_train_config(
            train_dataloader=mock_dataloader,
            arguments=func_args, config=updated_config,
        )

        assert updated_config["train_cfg"]["val_interval"] == engine.default_config["val_interval"]


    def test_update_config(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        engine = MMXEngine(work_dir=tmp_dir_path)

        # Test with model is None argument
        model = mocker.Mock()
        config, _ = engine._update_config({}, model=None)
        assert not hasattr(config, "model")
        assert config.default_scope == "mmengine"

        # Test with invalid argument
        model = mocker.Mock()
        config, _ = engine._update_config({}, invalid="test")
        assert not hasattr(config, "invalid")

        # Test with model argument
        model = mocker.Mock()
        config, _ = engine._update_config({"model": model})
        assert config["model"] == model

        # Test with model (Module) argument
        model = MockModel()
        config, _ = engine._update_config({"model": model})
        assert config["model"] == model

        # Test with train_dataloader argument
        train_dataloader = mocker.Mock()
        config, _ = engine._update_config({"train_dataloader": train_dataloader})
        assert config["train_dataloader"] == train_dataloader

        # Test with val_dataloader argument
        val_dataloader = mocker.Mock()
        config, _ = engine._update_config({"val_dataloader": val_dataloader})
        assert config["val_dataloader"] == val_dataloader
        assert "val_cfg" in config
        assert "val_evaluator" in config
        assert config["val_evaluator"] is not None

        # Test with test_dataloader argument
        test_dataloader = mocker.Mock()
        config, _ = engine._update_config({"test_dataloader": test_dataloader})
        assert config["test_dataloader"] == test_dataloader
        assert "test_cfg" in config
        assert "test_evaluator" in config
        assert config["test_evaluator"] is not None

        # Test with param_scheduler argument
        param_scheduler = {"foo": "bar"}
        config, _ = engine._update_config({"param_scheduler": param_scheduler})
        assert config["param_scheduler"] == param_scheduler

        # Test with custom_hooks argument
        custom_hooks = {"foo": "bar"}
        config, _ = engine._update_config({"custom_hooks": custom_hooks})
        assert config["custom_hooks"] == custom_hooks

        # Test with precision argument
        mocker.patch("otx.v2.adapters.torch.mmengine.engine.get_device", return_value="cpu")
        config, _ = engine._update_config({"train_dataloader": train_dataloader})
        assert config["optim_wrapper"]["type"] == "OptimWrapper"
        mocker.patch("otx.v2.adapters.torch.mmengine.engine.get_device", return_value="cuda")
        config, _ = engine._update_config({"precision": "fp16", "train_dataloader": train_dataloader})
        assert config["optim_wrapper"]["type"] == "AmpOptimWrapper"
        assert config["optim_wrapper"]["dtype"] == "fp16"

        # Test with seed & deterministic argument
        config, _ = engine._update_config({"seed": 123, "deterministic": True})
        assert config["randomness"]["seed"] == 123
        assert config["randomness"]["deterministic"]

        # Test with default_hooks argument
        default_hooks = {"foo": "bar"}
        config, _ = engine._update_config({"default_hooks": default_hooks})
        assert config["default_hooks"] == default_hooks

        # Test with visualizer argument
        visualizer = {"foo": "bar"}
        config, _ = engine._update_config({"visualizer": visualizer})
        assert config["visualizer"]["foo"] == "bar"


    def test_train(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.mmengine.engine.Path.glob", return_value=["test1.pth"])
        mocker.patch("otx.v2.adapters.torch.mmengine.engine.Path.unlink")
        mocker.patch("otx.v2.adapters.torch.mmengine.engine.shutil.copy")
        engine = MMXEngine(work_dir=tmp_dir_path)
        mock_runner = mocker.Mock
        engine.registry.register_module(name="Runner", module=mock_runner, force=True)
        mock_model = mocker.Mock()
        mock_dataloader = mocker.Mock()
        engine.train(model=mock_model, train_dataloader=mock_dataloader, checkpoint="test.pth")
        engine.runner.train.assert_called_once_with()
        engine.runner.load_checkpoint.assert_called_once_with("test.pth")

        engine.train(model=mock_model, train_dataloader=mock_dataloader, checkpoint="test.pth", max_iters=3, max_epochs=None)
        engine.runner.train.assert_called_once_with()
        engine.runner.load_checkpoint.assert_called_once_with("test.pth")

        mock_registry = mocker.Mock()
        mock_registry.get.return_value = None
        engine.registry = mock_registry
        with pytest.raises(ModuleNotFoundError):
            engine.train(max_epochs=3)

    def test_validate(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        engine = MMXEngine(work_dir=tmp_dir_path)
        mock_runner = mocker.Mock
        engine.registry.register_module(name="Runner", module=mock_runner, force=True)
        mock_model = mocker.Mock()
        mock_dataloader = mocker.Mock()
        engine.validate(model=mock_model, val_dataloader=mock_dataloader, checkpoint="test.pth")
        engine.runner.val.assert_called_once_with()
        engine.runner.load_checkpoint.assert_called_once_with("test.pth")

        engine.validate(precision="fp16")
        assert engine.runner._experiment_name.startswith("otx_validate_")

        engine = MMXEngine(work_dir=tmp_dir_path)
        mock_registry = mocker.Mock()
        mock_registry.get.return_value = None
        engine.registry = mock_registry
        with pytest.raises(ModuleNotFoundError):
            engine.validate(precision="fp16")

    def test_test(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        engine = MMXEngine(work_dir=tmp_dir_path)
        mock_runner = mocker.Mock
        engine.registry.register_module(name="Runner", module=mock_runner, force=True)
        mock_model = mocker.Mock()
        mock_dataloader = mocker.Mock()
        engine.test(model=mock_model, test_dataloader=mock_dataloader, checkpoint="test.pth")
        engine.runner.test.assert_called_once_with()
        engine.runner.load_checkpoint.assert_called_once_with("test.pth")

        engine.test(precision="fp16")
        assert engine.runner._experiment_name.startswith("otx_test_")

        engine = MMXEngine(work_dir=tmp_dir_path)
        mock_registry = mocker.Mock()
        mock_registry.get.return_value = None
        engine.registry = mock_registry
        with pytest.raises(ModuleNotFoundError):
            engine.test(precision="fp16")

    def test_export(self, mocker: MockerFixture, tmp_dir_path: Path, monkeypatch: MonkeyPatch) -> None:
        mock_exporter = mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.exporter.Exporter")
        mocker.patch("otx.v2.adapters.torch.mmengine.engine.Config.fromfile", return_value=mocker.MagicMock())
        engine = MMXEngine(work_dir=tmp_dir_path)
        engine.export(model="model_path", checkpoint="test.pth")
        mock_exporter.assert_called_once()

        engine.export(model=Config({}), checkpoint="test.pth")
        mock_exporter.assert_called()

        engine.export(model=MockModel())
        mock_exporter.assert_called()

        engine.dumped_config = {"model": {"type": "test"}}
        engine.export()
        mock_exporter.assert_called()

        engine.dumped_config = {"model": Config({"type": "test"})}
        engine.export()
        mock_exporter.assert_called()

        mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.utils.deploy_cfg_utils.patch_input_shape")
        mocker.patch("mmdeploy.utils.load_config", return_value=[{"backend_config": {}}])
        engine.export(deploy_config={}, input_shape=None)
        mock_exporter.assert_called()

        mocker.patch("mmdeploy.utils.load_config")
        mocker.patch("mmdeploy.utils.get_ir_config")
        mocker.patch("mmdeploy.utils.get_backend_config")
        mocker.patch("mmdeploy.utils.get_codebase_config")
        engine.export(deploy_config={})
        mock_exporter.assert_called()

        engine.dumped_config = {}
        with pytest.raises(ValueError, match="Not fount target model."):
            engine.export(model=None)

        with pytest.raises(NotImplementedError):
            engine.export(model=mocker.MagicMock())

        monkeypatch.setattr("otx.v2.adapters.torch.mmengine.engine.AVAILABLE", False)
        with pytest.raises(ModuleNotFoundError):
            engine.export(model="model_path", checkpoint="test.pth")

