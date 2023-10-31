"""OTX adapters.torch.lightning.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING, Union

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.optim import Optimizer

from otx.v2.api.core.engine import Engine
from otx.v2.api.utils import set_tuple_constructor
from otx.v2.api.utils.importing import get_all_args, get_default_args

from .registry import LightningRegistry

if TYPE_CHECKING:
    from pytorch_lightning.core.datamodule import LightningDataModule
    from pytorch_lightning.loggers.logger import Logger
    from pytorch_lightning.trainer.connectors.accelerator_connector import (
        _PRECISION_INPUT,
    )
    from pytorch_lightning.utilities.types import EVAL_DATALOADERS
    from torch.utils.data import DataLoader

PREDICT_FORMAT = Union[str, Path, np.ndarray]


class LightningEngine(Engine):
    """Lightning engine using PyTorch and PyTorch Lightning."""

    def __init__(
        self,
        work_dir: str | Path | None = None,
        config: str | dict | None = None,
        task: str = "classification",
    ) -> None:
        """Initialize the Lightning engine.

        Args:
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
            config (Optional[Union[str, dict]], optional): The configuration for the engine. Defaults to None.
            task (str, optional): The task to perform. Defaults to "classification".
        """
        super().__init__(work_dir=work_dir)
        self.trainer: Trainer
        self.trainer_config: dict = {}
        self.latest_model = {"model": None, "checkpoint": None}
        self.task = task
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
            self.trainer_config["deterministic"] = deterministic
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
    ) -> list[Logger] | Logger | None:
        """Update the logger and logs them to the console and any other configured loggers.

        Args:
            logger(list[Logger] | Logger | bool | None): Input of loggers
            target_path(str | None): logger's target output path

        Returns:
            list[Logger] | Logger | None: Updated loggers.
        """

    def _update_callbacks(
        self,
        callbacks: list[pl.Callback] | pl.Callback | DictConfig | None = None,
        mode: str | None = None,
    ) -> list[pl.Callback] | pl.Callback | None:
        """Update the list of callbacks to be executed during training and validation.

        Args:
            callbacks(list[pl.Callback] | pl.Callback | DictConfig | None): Input of callbacks
            mode(bool): Current Running mode status

        Returns:
            list[pl.Callback] | pl.Callback | None: Updated callbacks.
        """

    def _load_checkpoint(
        self,
        model: torch.nn.Module | pl.LightningModule,
        checkpoint: str | Path,
    ) -> None:
        if isinstance(model, pl.LightningModule):
            model = model.load_from_checkpoint(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location=model.device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)

    def _set_device(self, device: str | None = None) -> None:
        # Set accelerator
        accelerator = self.trainer_config.pop("accelerator", "auto")
        if device is not None:
            self.trainer_config["accelerator"] = device
        else:
            self.trainer_config["accelerator"] = accelerator

        # Set number of devices
        self.trainer_config["devices"] = self.trainer_config.get("devices", 1)

    def train(
        self,
        model: torch.nn.Module | pl.LightningModule,
        train_dataloader: DataLoader | LightningDataModule,
        val_dataloader: DataLoader | None = None,
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
        callbacks: list[pl.Callback] | pl.Callback | DictConfig | None = None,
        device: str | None = "auto",
        **kwargs,  # Trainer.__init__ arguments
    ) -> dict:
        """Train the given model using the provided data loaders and optimizer.

        Args:
            model (Union[torch.nn.Module, pl.LightningModule]): The model to train.
            train_dataloader (Union[DataLoader, LightningDataModule]): The data loader for training data.
            val_dataloader (Optional[DataLoader], optional): The data loader for validation data. Defaults to None.
            optimizer (Optional[Union[dict, Optimizer]], optional): The optimizer to use for training. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): The path to a checkpoint to load before training.
                Defaults to None.
            max_iters (Optional[int], optional): The maximum number of iterations to train for. Defaults to None.
            max_epochs (Optional[int], optional): The maximum number of epochs to train for. Defaults to None.
            distributed (Optional[bool], optional): Whether to use distributed training. Defaults to None.
            seed (Optional[int], optional): The random seed to use for training. Defaults to None.
            deterministic (Optional[bool], optional): Whether to use deterministic training. Defaults to None.
            precision (Optional[_PRECISION_INPUT], optional): The precision to use for training. Defaults to None.
            val_interval (Optional[int], optional): The interval at which to run validation. Defaults to None.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in train.
            callbacks (list[pl.Callback] | pl.Callback | DictConfig | None, optional): callbacks to use in train.
            device (str | None, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
            **kwargs: Additional arguments to pass to the Trainer constructor.

        Returns:
            dict: A dictionary containing the trained model and the path to the saved checkpoint.
        """
        _ = distributed
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
            mode = "train_val" if val_dataloader is not None else "train"
            callbacks = self._update_callbacks(callbacks=callbacks, mode=mode)
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
        model: torch.nn.Module | pl.LightningModule | None = None,
        val_dataloader: DataLoader | dict | None = None,
        checkpoint: str | Path | None = None,
        precision: _PRECISION_INPUT | None = None,
        logger: list[Logger] | Logger | bool | None = None,
        callbacks: list[pl.Callback] | pl.Callback | DictConfig | None = None,
        device: str | None = "auto",
        **kwargs,
    ) -> dict:
        """Run validation on the given model using the provided validation dataloader and checkpoint.

        Args:
            model (Optional[Union[torch.nn.Module, pl.LightningModule]]): The model to validate.
                If not provided, the latest model will be used.
            val_dataloader (Optional[Union[DataLoader, dict]]): The validation dataloader.
            checkpoint (Optional[Union[str, Path]]): The checkpoint to use for validation.
                If not provided, the latest checkpoint will be used.
            precision (Optional[_PRECISION_INPUT]): The precision to use for validation.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in validate.
            callbacks (list[pl.Callback] | pl.Callback | DictConfig | None, optional): callbacks to use in validate.
            device (str | None, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
            **kwargs: Additional keyword arguments to pass to the method.

        Returns:
            dict: The validation metric (data_class or dict).
        """
        update_check = self._update_config(func_args={"precision": precision}, **kwargs)

        datamodule = self.trainer_config.pop("datamodule", None)
        if model is None:
            model = self.latest_model.get("model")
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint")

        if not hasattr(self, "trainer") or update_check or device:
            self._set_device(device=device)
            callbacks = self._update_callbacks(callbacks=callbacks)
            logger = self._update_logger(logger=logger, target_path=f"{self.timestamp}_val")
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                **self.trainer_config,
            )

        if model is not None and checkpoint is not None:
            self._load_checkpoint(model, checkpoint)
        return self.trainer.validate(
            model=model,
            dataloaders=val_dataloader,
            ckpt_path=checkpoint,
            datamodule=datamodule,
        )

    def test(
        self,
        model: torch.nn.Module | pl.LightningModule | None = None,
        test_dataloader: DataLoader | None = None,
        checkpoint: str | Path | None = None,
        precision: _PRECISION_INPUT | None = None,
        logger: list[Logger] | Logger | bool | None = None,
        callbacks: list[pl.Callback] | pl.Callback | DictConfig | None = None,
        device: str | None = "auto",
        **kwargs,
    ) -> dict:
        """Test the given model on the provided test dataloader.

        Args:
            model (Optional[Union[torch.nn.Module, pl.LightningModule]]): The model to test.
                If not provided, the latest model will be used.
            test_dataloader (Optional[DataLoader]): The dataloader to use for testing.
            checkpoint (Optional[Union[str, Path]]): The checkpoint to use for testing.
                If not provided, the latest checkpoint will be used.
            precision (Optional[_PRECISION_INPUT]): The precision to use for testing.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in test.
            callbacks (list[pl.Callback] | pl.Callback | DictConfig | None, optional): callbacks to use in test.
            device (str | None, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
            **kwargs: Additional keyword arguments to pass to the method.

        Returns:
            dict: The test results as a dictionary.
        """
        update_check = self._update_config(func_args={"precision": precision}, **kwargs)
        if model is None:
            model = self.latest_model.get("model", None)
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)

        if not hasattr(self, "trainer") or update_check:
            self._set_device(device=device)
            callbacks = self._update_callbacks(callbacks=callbacks)
            logger = self._update_logger(logger=logger, target_path=f"{self.timestamp}_test")
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                **self.trainer_config,
            )

        if model is not None and checkpoint is not None:
            self._load_checkpoint(model, checkpoint)
        return self.trainer.test(
            model=model,
            dataloaders=[test_dataloader],
        )

    def predict(
        self,
        model: torch.nn.Module | pl.LightningModule | None = None,
        img: PREDICT_FORMAT | (EVAL_DATALOADERS | LightningDataModule) | None = None,
        checkpoint: str | Path | None = None,
        logger: list[Logger] | Logger | bool | None = None,
        callbacks: list[pl.Callback] | pl.Callback | DictConfig | None = None,
        device: str | None = "auto",  # ["auto", "cpu", "gpu", "cuda"]
    ) -> list:
        """Run inference on the given model and input data.

        Args:
            model (Optional[Union[torch.nn.Module, pl.LightningModule]]): The model to use for inference.
            img (Optional[Union[PREDICT_FORMAT, LightningDataModule]]): The input data to run inference on.
            checkpoint (Optional[Union[str, Path]]): The path to the checkpoint file to use for inference.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in prediction.
            callbacks (list[pl.Callback] | pl.Callback | DictConfig | None, optional): callbacks to use in prediction.
            device (str | None, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")

        Returns:
            list: The output of the inference.
        """
        if model is None:
            model = self.latest_model.get("model", None)
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)

        if not hasattr(self, "trainer"):
            self._set_device(device=device)
            callbacks = self._update_callbacks(callbacks=callbacks, mode="predict")
            logger = self._update_logger(logger=logger, target_path=f"{self.timestamp}_predict")
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                **self.trainer_config,
            )

        if model is not None and checkpoint is not None:
            self._load_checkpoint(model, checkpoint)
        # Lightning Inferencer
        return self.trainer.predict(
            model=model,
            dataloaders=[img],
        )

    def export(
        self,
        model: torch.nn.Module | pl.LightningModule | None = None,  # Module with _config OR Model Config OR config-file
        checkpoint: str | Path | None = None,
        precision: _PRECISION_INPUT | None = None,
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        device: str | None = None,
        input_shape: tuple[int, int] | None = None,
    ) -> dict:
        """Export the model to a specified format.

        Args:
            model (Optional[Union[torch.nn.Module, pl.LightningModule]]): The model to export.
            checkpoint (Optional[Union[str, Path]]): The checkpoint to use for exporting the model.
            precision (Optional[_PRECISION_INPUT]): The precision to use for exporting the model.
            export_type (str): The type of export to perform. Can be "ONNX" or "OPENVINO".
            input_shape (Optional[Tuple[int, int]]): The input shape to use for exporting the model.
            device (str | None, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps")

        Returns:
            dict: A dictionary containing the exported model(s).
        """
        # Set input_shape (input_size)
        _model = self.latest_model.get("model", None) if model is None else model
        model_config = self.config.get("model", {})
        height, width = model_config.get("input_size", (256, 256))
        if input_shape is not None:
            height, width = input_shape

        # Set device
        if device is None or device == "auto":
            device = getattr(model, "device", None)

        export_dir = self.work_dir / f"{self.timestamp}_export"
        export_dir.mkdir(exist_ok=True, parents=True)

        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)
        if _model is not None and checkpoint is not None:
            self._load_checkpoint(_model, checkpoint)

        # Torch to onnx
        onnx_dir = export_dir / "onnx"
        onnx_dir.mkdir(exist_ok=True, parents=True)
        onnx_model = str(onnx_dir / "onnx_model.onnx")
        torch.onnx.export(
            model=_model,
            args=torch.zeros((1, 3, height, width)).to(device),
            f=onnx_model,
            opset_version=11,
        )

        results: dict = {"outputs": {}}
        results["outputs"]["onnx"] = onnx_model

        if export_type.upper() == "OPENVINO":
            # ONNX to IR
            ir_dir = export_dir / "openvino"
            ir_dir.mkdir(exist_ok=True, parents=True)
            optimize_command = [
                "mo",
                "--input_model",
                onnx_model,
                "--output_dir",
                str(ir_dir),
                "--model_name",
                "openvino",
            ]
            if precision in ("16", 16, "fp16"):
                optimize_command.append("--compress_to_fp16")
            _ = run(args=optimize_command, check=False)
            bin_file = Path(ir_dir) / "openvino.bin"
            xml_file = Path(ir_dir) / "openvino.xml"
            if bin_file.exists() and xml_file.exists():
                results["outputs"]["bin"] = str(Path(ir_dir) / "openvino.bin")
                results["outputs"]["xml"] = str(Path(ir_dir) / "openvino.xml")
            else:
                msg = "OpenVINO Export failed."
                raise RuntimeError(msg)
        return results
