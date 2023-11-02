# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import pytorch_lightning as pl
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
        mock_model.callbacks = []
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
        mock_model.callbacks = []
        engine.validate(model=mock_model, val_dataloader=mock_dataloader)

        engine.trainer.validate.assert_called_once_with(model=mock_model, dataloaders=mock_dataloader, ckpt_path=None, datamodule=None)

    def test_test(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.lightning.engine.LightningEngine._update_config", return_value=True)
        mock_trainer = mocker.patch("otx.v2.adapters.torch.lightning.engine.Trainer")
        mock_trainer.return_value.test.return_value = {}
        mock_trainer.return_value.save_checkpoint.return_value = None
        engine = LightningEngine(work_dir=tmp_dir_path)
        mock_model = mocker.Mock()
        mock_model.callbacks = []
        mock_dataloader = mocker.Mock()

        engine.test(model=mock_model, test_dataloader=mock_dataloader)
        engine.trainer.test.assert_called_once_with(model=mock_model, dataloaders=[mock_dataloader], ckpt_path=None)

        engine.test(test_dataloader=mock_dataloader, checkpoint="test.pth")
        engine.trainer.test.assert_called_with(model=None, dataloaders=[mock_dataloader], ckpt_path="test.pth")

    def test_predict(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.lightning.engine.LightningEngine._update_config", return_value=True)
        mock_trainer = mocker.patch("otx.v2.adapters.torch.lightning.engine.Trainer")
        mock_trainer.return_value.predict.return_value = []
        mock_trainer.return_value.save_checkpoint.return_value = None
        engine = LightningEngine(work_dir=tmp_dir_path)
        mock_model = mocker.Mock()
        img = mocker.Mock()
        mock_model.callbacks = []
        engine.predict(model=mock_model, img=img)
        mock_trainer.return_value.predict.assert_called_once_with(model=mock_model, dataloaders=[img], ckpt_path=None)

        engine.predict(img=img, checkpoint="test.pth")
        mock_trainer.return_value.predict.assert_called_with(model=None, dataloaders=[img], ckpt_path="test.pth")

    def test_export(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        engine = LightningEngine(work_dir=tmp_dir_path)

        mock_model = mocker.Mock()
        mock_model.__class__.return_value = pl.LightningModule
        results = engine.export(model=mock_model, checkpoint="test.pth", precision="16")

        export_dir = tmp_dir_path / f"{engine.timestamp}_export"

        mock_model.export.assert_called_once_with(
            export_dir=export_dir,
            export_type="OPENVINO",
            precision="16",
        )
        assert results == mock_model.export.return_value

        with pytest.raises(NotImplementedError):
            engine.export(model=None)
