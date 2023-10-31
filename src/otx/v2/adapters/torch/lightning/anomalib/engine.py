"""OTX adapters.torch.lightning.anomalib.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.loggers import AnomalibTensorBoardLogger
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader

from otx.v2.adapters.torch.lightning.engine import LightningEngine

from .registry import AnomalibRegistry

if TYPE_CHECKING:
    import pytorch_lightning as pl
    import torch
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
        work_dir: str | Path | None = None,
        config: str | dict | None = None,
        task: str = "classification",
    ) -> None:
        """Initialize the Anomalib engine.

        Args:
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
            config (Optional[Union[str, dict]], optional): The configuration for the engine. Defaults to None.
            task (str, optional): The task to perform. Defaults to "classification".
        """
        super().__init__(work_dir=work_dir, config=config, task=task)
        self.registry = AnomalibRegistry()

    def _update_logger(
        self,
        logger: list[Logger] | Logger | bool | None = None,
        target_path: str | None = None,
    ) -> list[Logger] | Logger | None:
        """Update the logger and logs them to the console or use AnomalibTensorBoardLogger.

        Args:
            logger(list[Logger] | Logger | bool | None): Input of loggers
            target_path(str | None): logger's target output path

        Returns:
            list[Logger] | Logger | None: Updated loggers.
        """
        self.trainer_config.pop("logger", None)
        if logger is not None:
            if isinstance(logger, list):
                return logger
            if isinstance(logger, Logger):
                return [logger]
        return [AnomalibTensorBoardLogger(save_dir=self.work_dir, name=target_path)]

    def _update_callbacks(
        self,
        callbacks: list[pl.Callback] | pl.Callback | None = None,
        mode: str | None = None,
    ) -> list[pl.Callback] | pl.Callback | None:
        """Update the list of callbacks to be executed during training and validation.

        Args:
            callbacks(list[pl.Callback] | pl.Callback | None): Input of callbacks
            mode(bool): Current Running mode status

        Returns:
            list[pl.Callback] | pl.Callback | None: Updated callbacks.
        """
        _ = mode
        if callbacks is not None:
            if isinstance(callbacks, list):
                return callbacks
            return [callbacks]
        metrics = self.trainer_config.pop("metrics", None)
        if metrics is None:
            metrics = self.config.get("metrics", {})
        metric_threshold = metrics.get("threshold", {})
        return [
            MinMaxNormalizationCallback(),
            MetricsConfigurationCallback(
                task=self.task,
                image_metrics=metrics.get("image", None),
                pixel_metrics=metrics.get("pixel", None),
            ),
            PostProcessingConfigurationCallback(
                normalization_method=NormalizationMethod.MIN_MAX,
                threshold_method=ThresholdMethod.ADAPTIVE,
                manual_image_threshold=metric_threshold.get("manual_image", None),
                manual_pixel_threshold=metric_threshold.get("manual_pixel", None),
            ),
        ]


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
        logger: list[Logger] | Logger | None = None,
        callbacks: list[pl.Callback] | pl.Callback | None = None,
        **kwargs,  # Trainer.__init__ arguments
    ) -> dict:
        """Trains the given model using the provided data loaders and optimizer.

        Args:
            model: The PyTorch Lightning module to train.
            train_dataloader: The data loader for training data.
            val_dataloader: The data loader for validation data (optional).
            optimizer: The optimizer to use for training (optional).
            checkpoint: The path to save checkpoints during training (optional).
            max_iters: The maximum number of iterations to train for (optional).
            max_epochs: The maximum number of epochs to train for (optional).
            distributed: Whether to use distributed training (optional).
            seed: The random seed to use for training (optional).
            deterministic: Whether to use deterministic training (optional).
            precision: The precision to use for training (optional).
            val_interval: The number of training iterations between validation checks (optional).
            **kwargs: Additional arguments to pass to the PyTorch Lightning Trainer.

        Returns:
            A dictionary containing the training results.
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
            **kwargs,
        )

    def predict(
        self,
        model: torch.nn.Module | pl.LightningModule | None = None,
        img: PREDICT_FORMAT | (EVAL_DATALOADERS | LightningDataModule) | None = None,
        checkpoint: str | Path | None = None,
        device: list | None = None,  # ["auto", "cpu", "gpu", "cuda"]
        logger: list[Logger] | Logger | None = None,
        callbacks: list[pl.Callback] | pl.Callback | None = None,
    ) -> list:
        """Run inference on the given model and input data.

        Args:
            model (Optional[Union[torch.nn.Module, pl.LightningModule]]): The model to use for inference.
            img (Optional[Union[PREDICT_FORMAT, LightningDataModule]]): The input data to run inference on.
            checkpoint (Optional[Union[str, Path]]): The path to the checkpoint file to use for inference.
            device (Optional[list]): The device to use for inference. Can be "auto", "cpu", "gpu", or "cuda".

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
        # Lightning Inferencer
        return super().predict(
            model=model,
            img=dataloader,
            checkpoint=checkpoint,
            device=device,
            logger=logger,
            callbacks=callbacks,
        )
