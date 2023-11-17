"""Engine API for OTX lightning adapter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pytorch_lightning as pl
import yaml
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from otx.v2.adapters.torch.lightning.modules.models.base_model import BaseOTXLightningModel
from otx.v2.api.core.engine import Engine
from otx.v2.api.entities.task_type import TaskType
from otx.v2.api.utils import set_tuple_constructor
from otx.v2.api.utils.importing import get_all_args, get_default_args

from .registry import LightningRegistry

if TYPE_CHECKING:
    import torch
    from pytorch_lightning.core.datamodule import LightningDataModule
    from pytorch_lightning.loggers.logger import Logger
    from pytorch_lightning.trainer.connectors.accelerator_connector import (
        _PRECISION_INPUT,
    )
    from pytorch_lightning.utilities.types import EVAL_DATALOADERS

PREDICT_FORMAT = Union[str, Path, np.ndarray]


class LightningEngine(Engine):
    """Lightning engine using PyTorch and PyTorch Lightning."""

    def __init__(
        self,
        task: TaskType,
        work_dir: str | Path | None = None,
        config: str | dict | None = None,
    ) -> None:
        """Initialize the Lightning engine.

        Args:
            work_dir (str | Path | None, optional): The working directory for the engine. Defaults to None.
            config (str | dict | None, optional): The configuration for the engine. Defaults to None.
            task (str, optional): The task to perform. Defaults to "visual_prompting".
        """
        super().__init__(work_dir=work_dir, task=task)
        self.trainer: Trainer
        self.trainer_config: dict = {}
        self.latest_model: dict[str, torch.nn.Module | str | None] = {"model": None, "checkpoint": None}
        self.config = self._initial_config(config)
        if hasattr(self, "work_dir"):
            self.config.default_root_dir = self.work_dir
        self.registry = LightningRegistry()

    def _initial_config(self, config: str | dict | None = None) -> DictConfig:
        if isinstance(config, str) and config.endswith(".yaml"):
            set_tuple_constructor()
            with Path(config).open() as f:
                config = yaml.safe_load(f)
        elif config is None:
            config = {}
        return DictConfig(config)

    def _update_config(
        self,
        func_args: dict,
        **kwargs,
    ) -> bool:
        """Updates the trainer configuration with the given arguments.

        Args:
            func_args (dict): Dictionary containing function arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            bool: True if the configuration was updated, False otherwise.
        """
        update_check = not all(value is None for value in func_args.values()) or not all(
            value is None for value in kwargs.values()
        )
        if not self.trainer_config:
            self.trainer_config = self.config.get("trainer", {})
        for key, value in kwargs.items():
            self.trainer_config[key] = value

        precision = func_args.get("precision", None)
        max_epochs = func_args.get("max_epochs", None)
        max_iters = func_args.get("max_iters", None)
        seed = func_args.get("seed", None)
        deterministic = func_args.get("deterministic", None)
        val_interval = func_args.get("val_interval", None)
        if precision is not None:
            self.trainer_config["precision"] = precision
        if max_epochs is not None:
            self.trainer_config["max_epochs"] = max_epochs
            self.trainer_config["max_steps"] = -1
        elif max_iters is not None:
            self.trainer_config["max_epochs"] = None
            self.trainer_config["max_steps"] = max_iters
        if seed is not None:
            seed_everything(seed=seed)
        if deterministic is not None:
            self.trainer_config["deterministic"] = "warn" if deterministic else deterministic
        if val_interval is not None:
            # Validation Interval in Trainer -> val_check_interval
            self.trainer_config["val_check_interval"] = val_interval

        # Check Config Default is not None
        trainer_default_args = get_default_args(Trainer.__init__)
        for not_none_arg, default_value in trainer_default_args:
            if self.trainer_config.get(not_none_arg) is None:
                self.trainer_config[not_none_arg] = default_value
        # Last Check for Trainer.__init__
        trainer_arg_list = get_all_args(Trainer.__init__)
        removed_key = [config_key for config_key in self.trainer_config if config_key not in trainer_arg_list]
        if removed_key:
            for config_key in removed_key:
                self.trainer_config.pop(config_key)

        return update_check

    def _update_logger(
        self,
        logger: list[Logger] | Logger | bool | None = None,
        target_path: str | None = None,
    ) -> list[Logger] | None:
        """Update the logger and logs them to the console and any other configured loggers.

        Args:
            logger(list[Logger] | Logger | bool | None, optional): Input of loggers
            target_path(str | None, optional): logger's target output path

        Returns:
            list[Logger] | None: Updated loggers.
        """
        self.trainer_config.pop("logger", None)
        logger_list = [CSVLogger(save_dir=self.work_dir, name=target_path, version=self.timestamp)]
        if isinstance(logger, list):
            logger_list.extend(logger)
        elif logger is False:
            return None
        elif logger is not None:
            logger_list.append(logger)
        return logger_list

    def _set_device(self, device: str = "auto") -> None:
        """Sets the device for the trainer.

        Args:
            device (str, optional): The device to use. Defaults to "auto".
        """
        # Set accelerator
        self.trainer_config["accelerator"] = device

        # Set number of devices
        self.trainer_config["devices"] = self.trainer_config.get("devices", 1)

    def train(
        self,
        model: BaseOTXLightningModel | pl.LightningModule,
        train_dataloader: DataLoader | LightningDataModule,
        val_dataloader: DataLoader | LightningDataModule | None = None,
        optimizer: dict | Optimizer | None = None,
        checkpoint: str | Path | None = None,
        max_iters: int | None = None,
        max_epochs: int | None = None,
        distributed: bool | None = None,
        seed: int | None = None,
        deterministic: bool | None = None,
        precision: _PRECISION_INPUT | None = None,
        val_interval: int | None = None,
        logger: list[Logger] | Logger | bool | None = None,
        callbacks: list[pl.Callback] | None = None,
        device: str = "auto",
        **kwargs,  # Trainer.__init__ arguments
    ) -> dict:
        """Train the given model using the provided data loaders and optimizer.

        Args:
            model (BaseOTXLightningModel | pl.LightningModule): The model to train.
            train_dataloader (DataLoader | LightningDataModule): The data loader for training data.
            val_dataloader (DataLoader | LightningDataModule | None, optional): The data loader for validation data.
            optimizer (dict | Optimizer | None, optional): The optimizer to use for training. Defaults to None.
            checkpoint (str | Path | None, optional): The path to a checkpoint to load before training.
                Defaults to None.
            max_iters (int | None, optional): The maximum number of iterations to train for. Defaults to None.
            max_epochs (int | None, optional): The maximum number of epochs to train for. Defaults to None.
            distributed (bool | None, optional): Whether to use distributed training. Defaults to None.
            seed (int | None, optional): The random seed to use for training. Defaults to None.
            deterministic (bool | None, optional): Whether to use deterministic training. Defaults to None.
            precision (_PRECISION_INPUT | None, optional): The precision to use for training. Defaults to None.
            val_interval (int | None, optional): The interval at which to run validation. Defaults to None.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in train.
            callbacks (list[pl.Callback] | None, optional): callbacks to use in train.
            device (str, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
            **kwargs (Any): Additional arguments to pass to the Trainer constructor.

        Returns:
            dict: A dictionary containing the trained model and the path to the saved checkpoint.
        """
        _ = distributed
        callbacks = [] if callbacks is None else callbacks
        target_path = f"{self.timestamp}_train"
        train_args = {
            "max_iters": max_iters,
            "max_epochs": max_epochs,
            "seed": seed,
            "deterministic": deterministic,
            "precision": precision,
            "val_interval": val_interval,
        }
        update_check = self._update_config(func_args=train_args, **kwargs)
        datamodule = self.trainer_config.pop("datamodule", None)

        if not hasattr(self, "trainer") or update_check:
            self._set_device(device=device)
            if hasattr(model, "callbacks"):
                callbacks.extend(model.callbacks)
            logger = self._update_logger(logger=logger, target_path=target_path)
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                **self.trainer_config,
            )
        self.config["trainer"] = self.trainer_config

        if isinstance(optimizer, Optimizer):
            self.trainer.optimizers = optimizer

        self.trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            datamodule=datamodule,
            ckpt_path=checkpoint,
        )

        output_model_dir = self.work_dir / target_path / "models"
        self.trainer.save_checkpoint(output_model_dir / "weights.pth")
        results = {"model": model, "checkpoint": str(output_model_dir / "weights.pth")}
        self.latest_model = results
        return results

    def validate(
        self,
        model: BaseOTXLightningModel | pl.LightningModule | None = None,
        val_dataloader: DataLoader | LightningDataModule | None = None,
        checkpoint: str | Path | None = None,
        precision: _PRECISION_INPUT | None = None,
        logger: list[Logger] | Logger | bool | None = None,
        callbacks: list[pl.Callback] | None = None,
        device: str = "auto",
        **kwargs,
    ) -> dict:
        """Run validation on the given model using the provided validation dataloader and checkpoint.

        Args:
            model (BaseOTXLightningModel | pl.LightningModule | None, optional): The model to validate.
                If not provided, the latest model will be used.
            val_dataloader (DataLoader | LightningDataModule | None, optional): The validation dataloader.
            checkpoint (str | Path | None, optional): The checkpoint to use for validation.
                If not provided, the latest checkpoint will be used.
            precision (_PRECISION_INPUT | None, optional): The precision to use for validation.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in validate.
            callbacks (list[pl.Callback] | None, optional): callbacks to use in validate.
            device (str, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
            **kwargs (Any): Additional keyword arguments to pass to the method.

        Returns:
            dict: The validation metric (data_class or dict).
        """
        update_check = self._update_config(func_args={"precision": precision}, **kwargs)
        callbacks = callbacks if callbacks is not None else []
        datamodule = self.trainer_config.pop("datamodule", None)
        if model is None:
            model = self.latest_model.get("model")
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint")

        if not hasattr(self, "trainer") or update_check:
            self._set_device(device=device)
            if model is not None and hasattr(model, "callbacks"):
                callbacks.extend(model.callbacks)
            logger = self._update_logger(logger=logger, target_path=f"{self.timestamp}_val")
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                **self.trainer_config,
            )

        checkpoint = str(checkpoint) if checkpoint is not None else None
        return self.trainer.validate(
            model=model,
            dataloaders=val_dataloader,
            ckpt_path=checkpoint,
            datamodule=datamodule,
        )

    def test(
        self,
        model: BaseOTXLightningModel | pl.LightningModule | None = None,
        test_dataloader: DataLoader | LightningDataModule | None = None,
        checkpoint: str | Path | None = None,
        precision: _PRECISION_INPUT | None = None,
        logger: list[Logger] | Logger | bool | None = None,
        callbacks: list[pl.Callback] | None = None,
        device: str = "auto",
        **kwargs,
    ) -> dict:
        """Test the given model on the provided test dataloader.

        Args:
            model (BaseOTXLightningModel | pl.LightningModule | None, optional): The model to test.
                If not provided, the latest model will be used.
            test_dataloader (DataLoader | LightningDataModule | None, optional): The dataloader to use for testing.
            checkpoint (str | Path | None, optional): The checkpoint to use for testing.
                If not provided, the latest checkpoint will be used.
            precision ( _PRECISION_INPUT | None, optional): The precision to use for testing.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in test.
            callbacks (list[pl.Callback] | None, optional): callbacks to use in test.
            device (str, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
            **kwargs (Any): Additional keyword arguments to pass to the method.

        Returns:
            dict: The test results as a dictionary.
        """
        update_check = self._update_config(func_args={"precision": precision}, **kwargs)
        callbacks = callbacks if callbacks is not None else []
        if model is None:
            model = self.latest_model.get("model", None)
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)

        if not hasattr(self, "trainer") or update_check:
            self._set_device(device=device)
            if model is not None and hasattr(model, "callbacks"):
                callbacks.extend(model.callbacks)
            logger = self._update_logger(logger=logger, target_path=f"{self.timestamp}_test")
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                **self.trainer_config,
            )

        checkpoint = str(checkpoint) if checkpoint is not None else None
        return self.trainer.test(
            model=model,
            dataloaders=[test_dataloader],
            ckpt_path=checkpoint,
        )

    def predict(
        self,
        model: BaseOTXLightningModel | pl.LightningModule | None = None,
        img: PREDICT_FORMAT | (EVAL_DATALOADERS | LightningDataModule) | None = None,
        checkpoint: str | Path | None = None,
        logger: list[Logger] | Logger | bool | None = None,
        callbacks: list[pl.Callback] | None = None,
        device: str = "auto",
    ) -> list:
        """Run inference on the given model and input data.

        Args:
            model (BaseOTXLightningModel | pl.LightningModule | None, optional): The model to use for inference.
            img (PREDICT_FORMAT | (EVAL_DATALOADERS | LightningDataModule) | None, optional): The input data
                to run inference on.
            checkpoint (str | Path | None, optional): The path to the checkpoint file to use for inference.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in prediction.
            callbacks (list[pl.Callback] | None, optional): callbacks to use in prediction.
            device (str, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")

        Returns:
            list: The output of the inference.
        """
        dataloader = None
        # NOTE: It needs to be refactored in a more general way.
        if self.task.name.lower() == "visual_prompting" and isinstance(img, (str, Path)):
            from .modules.datasets.visual_prompting_dataset import VisualPromptInferenceDataset

            dataset_config = self.config.get("dataset", {})
            image_size = dataset_config.get("image_size", 1024)
            dataset = VisualPromptInferenceDataset(path=img, image_size=image_size)
            dataloader = DataLoader(dataset)
        if dataloader is None:
            dataloader = [img]

        callbacks = callbacks if callbacks is not None else []
        if model is None:
            model = self.latest_model.get("model", None)
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)

        if not hasattr(self, "trainer"):
            self._set_device(device=device)
            if model is not None and hasattr(model, "callbacks"):
                callbacks.extend(model.callbacks)
            logger = self._update_logger(logger=logger, target_path=f"{self.timestamp}_predict")
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                **self.trainer_config,
            )

        checkpoint = str(checkpoint) if checkpoint is not None else None
        # Lightning Inferencer
        return self.trainer.predict(
            model=model,
            dataloaders=dataloader,
            ckpt_path=checkpoint,
        )

    def export(
        self,
        model: BaseOTXLightningModel | pl.LightningModule | None = None,
        checkpoint: str | Path | None = None,
        precision: _PRECISION_INPUT | None = None,
        export_type: str = "OPENVINO",
    ) -> dict:
        """Export the model to a specified format.

        Args:
            model (BaseOTXLightningModel | pl.LightningModule | None, optional): The model to export.
            checkpoint (str | Path | None, optional): The checkpoint to use for exporting the model.
            precision (_PRECISION_INPUT | None, optional): The precision to use for exporting the model.
            export_type (str, optional): The type of export to perform. Can be "ONNX" or "OPENVINO".

        Returns:
            dict: A dictionary containing the exported model(s).
        """
        # Set input_shape (input_size)
        _model = self.latest_model.get("model", None) if model is None else model

        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint")
        if checkpoint is not None and isinstance(_model, pl.LightningModule):
            _model = _model.load_from_checkpoint(checkpoint)

        export_dir = self.work_dir / f"{self.timestamp}_export"
        if _model is not None and hasattr(_model, "export"):
            return _model.export(
                export_dir=export_dir,
                export_type=export_type,
                precision=precision,
            )
        msg = f"{_model} does not support export."
        raise NotImplementedError(msg)
