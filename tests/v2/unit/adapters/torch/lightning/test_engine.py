# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig
from otx.v2.adapters.torch.lightning.engine import LightningEngine
from pytest_mock.plugin import MockerFixture


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._config = DictConfig({})


class TestLightningEngine:
    def test_init(self, tmp_dir_path: Path) -> None:
        engine = LightningEngine(work_dir=tmp_dir_path)
        assert engine.work_dir == tmp_dir_path
        assert engine.registry.name == "lightning"
        assert engine.timestamp is not None

    def test_initial_config(self, tmp_dir_path: Path) -> None:
        engine = LightningEngine(work_dir=tmp_dir_path)
        assert isinstance(engine.config, DictConfig)

        config = {"foo": "bar"}
        engine = LightningEngine(work_dir=tmp_dir_path, config=config)
        assert isinstance(engine.config, DictConfig)
        assert hasattr(engine.config, "foo")
        assert engine.config.foo == "bar"

        config_file = tmp_dir_path / "config.yaml"
        config_file.write_text("foo : 'bar'")
        engine = LightningEngine(work_dir=tmp_dir_path, config=str(config_file))
        assert isinstance(engine.config, DictConfig)
        assert hasattr(engine.config, "foo")
        assert engine.config.foo == "bar"

    def test_update_config(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        engine = LightningEngine(work_dir=tmp_dir_path)

        # Test with invalid argument
        engine._update_config({}, invalid="test")
        assert not hasattr(engine.trainer_config, "invalid")

        # Test with max_epochs argument
        engine._update_config({"max_epochs": 3})
        assert engine.trainer_config["max_epochs"] == 3
        assert engine.trainer_config["max_steps"] == -1

        # Test with max_iters argument
        engine._update_config({"max_iters": 3})
        assert engine.trainer_config["max_epochs"] is None
        assert engine.trainer_config["max_steps"] == 3

        # Test with max_iters argument
        engine._update_config({"precision": "16"})
        assert engine.trainer_config["precision"] == "16"

        # Test with seed and deterministic argument
        mock_seed_everything = mocker.patch("otx.v2.adapters.torch.lightning.engine.seed_everything")
        engine._update_config({"seed": 123, "deterministic": True})
        assert engine.trainer_config["deterministic"]
        mock_seed_everything.assert_called_once_with(seed=123)

        # Test with val_interval argument
        engine._update_config({"val_interval": 3})
        assert engine.trainer_config["val_check_interval"] == 3

    def test_train(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.lightning.engine.LightningEngine._update_config", return_value=True)
        mock_trainer = mocker.patch("otx.v2.adapters.torch.lightning.engine.Trainer")
        mock_trainer.return_value.fit.return_value = None
        mock_trainer.return_value.save_checkpoint.return_value = None
        engine = LightningEngine(work_dir=tmp_dir_path)
        mock_model = mocker.Mock()
        mock_dataloader = mocker.Mock()
        results = engine.train(model=mock_model, train_dataloader=mock_dataloader, optimizer=mocker.Mock())
        engine.trainer.fit.assert_called_once_with(model=mock_model, train_dataloaders=mock_dataloader, val_dataloaders=None, datamodule=None, ckpt_path=None)
        output_model_dir = tmp_dir_path / f"{engine.timestamp}_train" / "models" / "weights.pth"
        engine.trainer.save_checkpoint.assert_called_once_with(output_model_dir)

        assert results["model"] == mock_model
        assert results["checkpoint"] == str(output_model_dir)

    def test_validate(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.lightning.engine.LightningEngine._update_config", return_value=True)
        mock_trainer = mocker.patch("otx.v2.adapters.torch.lightning.engine.Trainer")
        mock_trainer.return_value.validate.return_value = {}
        mock_trainer.return_value.save_checkpoint.return_value = None
        engine = LightningEngine(work_dir=tmp_dir_path)
        mock_model = mocker.Mock()
        mock_dataloader = mocker.Mock()
        engine.validate(model=mock_model, val_dataloader=mock_dataloader)

        engine.trainer.validate.assert_called_once_with(model=mock_model, dataloaders=mock_dataloader, ckpt_path=None, datamodule=None)

    def test_test(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.lightning.engine.LightningEngine._update_config", return_value=True)
        mock_trainer = mocker.patch("otx.v2.adapters.torch.lightning.engine.Trainer")
        mock_trainer.return_value.test.return_value = {}
        mock_trainer.return_value.save_checkpoint.return_value = None
        engine = LightningEngine(work_dir=tmp_dir_path)
        mock_model = mocker.Mock()
        mock_dataloader = mocker.Mock()

        engine.test(model=mock_model, test_dataloader=mock_dataloader)
        engine.trainer.test.assert_called_once_with(model=mock_model, dataloaders=[mock_dataloader])

        engine.test(test_dataloader=mock_dataloader)
        engine.trainer.test.assert_called_with(model=None, dataloaders=[mock_dataloader])

    def test_predict(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.lightning.engine.LightningEngine._update_config", return_value=True)
        mock_trainer = mocker.patch("otx.v2.adapters.torch.lightning.engine.Trainer")
        mock_trainer.return_value.predict.return_value = []
        mock_trainer.return_value.save_checkpoint.return_value = None
        engine = LightningEngine(work_dir=tmp_dir_path)
        mock_model = mocker.Mock()
        img = mocker.Mock()

        engine.predict(model=mock_model, img=img)
        mock_trainer.return_value.predict.assert_called_once_with(model=mock_model, dataloaders=[img])

        engine.predict(img=img)
        mock_trainer.return_value.predict.assert_called_with(model=None, dataloaders=[img])

    def test_export(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mock_export = mocker.patch("otx.v2.adapters.torch.lightning.engine.torch.onnx.export")
        mocker.patch("otx.v2.adapters.torch.lightning.engine.Path.mkdir")
        mocker.patch("otx.v2.adapters.torch.lightning.engine.Path.exists", return_value=True)
        mock_load = mocker.patch("otx.v2.adapters.torch.lightning.engine.torch.load")
        mock_load.return_value = {"model": {"state_dict": {}}}
        mock_run = mocker.patch("otx.v2.adapters.torch.lightning.engine.run")
        mock_zeros = mocker.patch("otx.v2.adapters.torch.lightning.engine.torch.zeros")
        mock_zeros.return_value.to.return_value = mocker.Mock()
        engine = LightningEngine(work_dir=tmp_dir_path)

        mock_model = mocker.Mock()
        results = engine.export(model=mock_model, checkpoint="test.pth", precision="16", device="cpu", input_shape=(321, 321))

        export_dir = tmp_dir_path / f"{engine.timestamp}_export"

        mock_export.assert_called_once_with(
            model=mock_model.model,
            args=mock_zeros.return_value.to.return_value,
            f=str(export_dir / "onnx" / "onnx_model.onnx"),
            opset_version=11,
        )
        mock_run.assert_called_once_with(
            args=[
                "mo",
                "--input_model",
                str(export_dir / "onnx" / "onnx_model.onnx"),
                "--output_dir",
                str(export_dir / "openvino"),
                "--model_name", "openvino",
                "--compress_to_fp16",
            ],
            check=False,
        )
        assert "outputs" in results
        assert "onnx" in results["outputs"]
        assert results["outputs"]["onnx"] == str(export_dir / "onnx" / "onnx_model.onnx")
        assert "bin" in results["outputs"]
        assert results["outputs"]["bin"] == str(export_dir / "openvino" / "openvino.bin")
        assert "xml" in results["outputs"]
        assert results["outputs"]["xml"] == str(export_dir / "openvino" / "openvino.xml")

        mock_load = mocker.patch("otx.v2.adapters.torch.lightning.engine.torch.load")
        results = engine.export(model=mock_model)
        mock_load.assert_not_called()

        mocker.patch("otx.v2.adapters.torch.lightning.engine.Path.exists", return_value=False)
        with pytest.raises(RuntimeError, match="OpenVINO Export failed."):
            engine.export(model=mock_model, checkpoint="test.pth", precision="16")
