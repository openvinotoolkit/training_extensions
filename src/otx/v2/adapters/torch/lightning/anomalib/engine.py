"""OTX adapters.torch.lightning.anomalib.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pytorch_lightning as pl
import torch
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.utils.loggers import AnomalibTensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader

from otx.v2.adapters.torch.lightning.engine import LightningEngine
from otx.v2.api.entities.task_type import TaskType

from .registry import AnomalibRegistry

if TYPE_CHECKING:
    from pytorch_lightning.trainer.connectors.accelerator_connector import (
        _PRECISION_INPUT,
    )
    from pytorch_lightning.utilities.types import EVAL_DATALOADERS
    from torch.optim import Optimizer

PREDICT_FORMAT = Union[str, Path, np.ndarray]


class AnomalibEngine(LightningEngine):
    """Anomalib engine using PyTorch and PyTorch Lightning."""

    def __init__(
        self,
        task: TaskType,
        work_dir: str | Path | None = None,
        config: str | dict | None = None,
    ) -> None:
        """Initialize the Anomalib engine.

        Args:
            task (TaskType): The task to perform.
            work_dir (str | Path | None, optional): The working directory for the engine. Defaults to None.
            config (str | dict | None, optional): The configuration for the engine. Defaults to None.
        """
        super().__init__(work_dir=work_dir, config=config, task=task)
        self.registry = AnomalibRegistry()

    def _update_logger(
        self,
        logger: list[Logger] | Logger | bool | None = None,
        target_path: str | None = None,
    ) -> list[Logger] | None:
        """Update the logger and logs them to the console or use AnomalibTensorBoardLogger.

        Args:
            logger(list[Logger] | Logger | bool | None, optional): Input of loggers
            target_path(str | None, optional): logger's target output path

        Returns:
            list[Logger] | None: Updated loggers.
        """
        self.trainer_config.pop("logger", None)
        if logger is not None:
            if isinstance(logger, list):
                return logger
            if isinstance(logger, Logger):
                return [logger]
        return [AnomalibTensorBoardLogger(save_dir=self.work_dir, name=target_path)]

    def _load_checkpoint(
        self,
        model: torch.nn.Module | pl.LightningModule,
        checkpoint: str | Path,
    ) -> None:
        """Loads a checkpoint for the given model.

        Args:
            model (torch.nn.Module | pl.LightningModule): The model to load the checkpoint for.
            checkpoint (str | Path): The path to the checkpoint file.

        Returns:
            None
        """
        if isinstance(model, pl.LightningModule):
            model = model.load_from_checkpoint(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location=model.device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)

    def train(
        self,
        model: torch.nn.Module | pl.LightningModule,
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
        **kwargs,
    ) -> dict:
        """Trains the given model using the provided data loaders and optimizer.

        Args:
            model (torch.nn.Module | pl.LightningModule): The PyTorch Lightning module to train.
            train_dataloader (DataLoader | LightningDataModule): The data loader for training data.
            val_dataloader (DataLoader | LightningDataModule | None, optional): The data loader for validation data.
            optimizer (dict | Optimizer | None, optional): The optimizer to use for training.
            checkpoint (str | Path | None, optional): The path to save checkpoints during training.
            max_iters (int | None, optional): The maximum number of iterations to train.
            max_epochs (int | None, optional): The maximum number of epochs to train.
            distributed (bool | None, optional): Whether to use distributed training.
            seed (int | None, optional): The random seed to use for training.
            deterministic (bool | None, optional): Whether to use deterministic training.
            precision (_PRECISION_INPUT | None, optional): The precision to use for training.
            val_interval (int | None, optional): The number of training iterations between validation checks.
            logger (list[Logger] | Logger | bool | None, optional): logger for training.
            callbacks (list[pl.Callback] | None, optional): callbacks to use in train.
            device (str, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"), Default to "auto".
            **kwargs (Any): Additional arguments to pass to the PyTorch Lightning Trainer.

        Returns:
            dict: A dictionary containing the training results.
        """
        kwargs["num_sanity_val_steps"] = kwargs.pop("num_sanity_val_steps", 0)
        return super().train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            checkpoint=checkpoint,
            max_iters=max_iters,
            max_epochs=max_epochs,
            distributed=distributed,
            seed=seed,
            deterministic=deterministic,
            precision=precision,
            val_interval=val_interval,
            logger=logger,
            callbacks=callbacks,
            device=device,
            **kwargs,
        )

    def test(
        self,
        model: torch.nn.Module | pl.LightningModule | None = None,
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
            model (torch.nn.Module | pl.LightningModule | None, optional): The model to test.
                If not provided, the latest model will be used.
            test_dataloader (DataLoader | LightningDataModule | None, optional): The dataloader to use for testing.
            checkpoint (str | Path | None, optional): The checkpoint to use for testing.
                If not provided, the latest checkpoint will be used.
            precision (_PRECISION_INPUT | None, optional): The precision to use for testing.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in test.
            callbacks (list[pl.Callback] | None, optional): callbacks to use in test.
            device (str, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"), Default to "auto".
            **kwargs (Any): Additional arguments to pass to the PyTorch Lightning Trainer.

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
        callbacks: list[pl.Callback] | None = None,
        device: str = "auto",
    ) -> list:
        """Run inference on the given model and input data.

        Args:
            model (torch.nn.Module | pl.LightningModule | None, optional): The model to use for inference.
            img (PREDICT_FORMAT | (EVAL_DATALOADERS | LightningDataModule) | None, optional): The input data
                to run inference on.
            checkpoint (str | Path | None, optional): The path to the checkpoint file to use for inference.
            logger (list[Logger] | Logger | bool | None, optional): Logger to use in prediction.
            callbacks (list[pl.Callback] | None, optional): callbacks to use in prediction.
            device (str, optional): Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"), Default to "auto".

        Returns:
            list: The output of the inference.
        """
        dataloader = None
        if isinstance(img, (str, Path)):
            dataset_config = self.config.get("dataset", {})
            transform_config = dataset_config.transform_config.eval if "transform_config" in dataset_config else None
            image_size = tuple(dataset_config.get("image_size", (256, 256)))
            center_crop = dataset_config.get("center_crop")
            if center_crop is not None:
                center_crop = tuple(center_crop)
            normalization = InputNormalizationMethod(dataset_config.get("normalization", "imagenet"))
            transform = get_transforms(
                config=transform_config,
                image_size=image_size,
                center_crop=center_crop,
                normalization=normalization,
            )
            dataset = InferenceDataset(path=img, image_size=image_size, transform=transform)
            dataloader = DataLoader(dataset)
        elif isinstance(img, (DataLoader, LightningDataModule)):
            dataloader = [img]
        if model is None:
            model = self.latest_model.get("model", None)
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)

        callbacks = callbacks if callbacks is not None else []
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

        if model is not None and checkpoint is not None:
            self._load_checkpoint(model, checkpoint)
        return self.trainer.predict(
            model=model,
            dataloaders=dataloader,
        )

    def export(
        self,
        model: torch.nn.Module | pl.LightningModule | None = None,
        checkpoint: str | Path | None = None,
        precision: _PRECISION_INPUT | None = None,
        export_type: str = "OPENVINO",
    ) -> dict[str, str | dict]:
        """Export the model to a specified format.

        Args:
            model (torch.nn.Module | pl.LightningModule | None, optional): The model to export.
            checkpoint (str | Path | None, optional): The checkpoint to use for exporting the model.
            precision (_PRECISION_INPUT | None, optional): The precision to use for exporting the model.
            export_type (str, optional): The type of export to perform. Can be "ONNX" or "OPENVINO".

        Returns:
            dict[str, str | dict]: A dictionary containing the exported model(s).
        """
        _model = self.latest_model.get("model", None) if model is None else model

        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint")
        if _model is not None and checkpoint is not None:
            self._load_checkpoint(_model, checkpoint)

        export_dir = self.work_dir / f"{self.timestamp}_export"
        if _model is not None and hasattr(_model, "export"):
            return _model.export(
                export_dir=export_dir,
                export_type=export_type,
                precision=precision,
            )
        msg = f"{_model} does not support export."
        raise NotImplementedError(msg)
