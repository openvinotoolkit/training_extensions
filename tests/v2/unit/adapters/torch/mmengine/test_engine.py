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

    def test_initial_config(self, tmp_dir_path: Path) -> None:
        engine = MMXEngine(work_dir=tmp_dir_path)
        assert isinstance(engine.config, Config)

        config = {"foo": "bar"}
        engine = MMXEngine(work_dir=tmp_dir_path, config=config)
        assert isinstance(engine.config, Config)
        assert hasattr(engine.config, "foo")
        assert engine.config.foo == "bar"

        config = Config({"foo": "bar"})
        engine = MMXEngine(work_dir=tmp_dir_path, config=config)
        assert isinstance(engine.config, Config)
        assert hasattr(engine.config, "foo")
        assert engine.config.foo == "bar"

        config_file = tmp_dir_path / "config.yaml"
        config_file.write_text("foo : 'bar'")
        engine = MMXEngine(work_dir=tmp_dir_path, config=str(config_file))
        assert isinstance(engine.config, Config)
        assert hasattr(engine.config, "foo")
        assert engine.config.foo == "bar"

    def test_update_config(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        engine = MMXEngine(work_dir=tmp_dir_path)

        # Test with model is None argument
        model = mocker.Mock()
        engine._update_config({}, model=None)
        assert not hasattr(engine.config, "model")

        # Test with invalid argument
        model = mocker.Mock()
        engine._update_config({}, invalid="test")
        assert not hasattr(engine.config, "invalid")

        # Test with model argument
        model = mocker.Mock()
        engine._update_config({"model": model})
        assert engine.config["model"] == model

        # Test with model (Module) argument
        model = MockModel()
        engine._update_config({"model": model})
        assert engine.config["model"] == model

        # Test with train_dataloader argument
        train_dataloader = mocker.Mock()
        engine._update_config({"train_dataloader": train_dataloader})
        assert engine.config["train_dataloader"] == train_dataloader

        # Test with val_dataloader argument
        val_dataloader = mocker.Mock()
        engine._update_config({"val_dataloader": val_dataloader})
        assert engine.config["val_dataloader"] == val_dataloader
        assert "val_cfg" in engine.config
        assert "val_evaluator" in engine.config
        assert engine.config["val_evaluator"]

        # Test with test_dataloader argument
        test_dataloader = mocker.Mock()
        engine._update_config({"test_dataloader": test_dataloader})
        assert engine.config["test_dataloader"] == test_dataloader

        # Test with param_scheduler argument
        param_scheduler = {"foo": "bar"}
        engine._update_config({"param_scheduler": param_scheduler})
        assert engine.config["param_scheduler"] == param_scheduler

        # Test with custom_hooks argument
        custom_hooks = {"foo": "bar"}
        engine._update_config({"custom_hooks": custom_hooks})
        assert engine.config["custom_hooks"] == custom_hooks

        # Test with precision argument
        mocker.patch("otx.v2.adapters.torch.mmengine.utils.runner_config.get_device", return_value="cpu")
        engine.config["optim_wrapper"] = None
        engine._update_config({}, train_dataloader=train_dataloader)
        assert engine.config["optim_wrapper"]["type"] == "OptimWrapper"
        mocker.patch("otx.v2.adapters.torch.mmengine.utils.runner_config.get_device", return_value="cuda")
        engine.config["optim_wrapper"] = None
        engine._update_config({"precision": "fp16"}, train_dataloader=train_dataloader)
        assert engine.config["optim_wrapper"]["type"] == "AmpOptimWrapper"
        assert engine.config["optim_wrapper"]["dtype"] == "fp16"

        # Test with seed & deterministic argument
        engine._update_config({"seed": 123, "deterministic": True})
        assert engine.config["randomness"]["seed"] == 123
        assert engine.config["randomness"]["deterministic"]

        # Test with default_hooks argument
        default_hooks = {"foo": "bar"}
        engine._update_config({"default_hooks": default_hooks})
        assert engine.config["default_hooks"] == default_hooks

        # Test with visualizer argument
        visualizer = {"foo": "bar"}
        engine._update_config({"visualizer": visualizer})
        assert engine.config["visualizer"]["foo"] == "bar"


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

        engine.train(model=mock_model, train_dataloader=mock_dataloader, checkpoint="test.pth", max_iters=3)
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

