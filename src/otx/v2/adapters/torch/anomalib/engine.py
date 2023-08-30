"""OTX adapters.torch.anomalib.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _PRECISION_INPUT,
)
from torch.utils.data import DataLoader

from otx.v2.api.core.engine import Engine

from .registry import AnomalibRegistry


class AnomalibEngine(Engine):
    def __init__(
        self,
        work_dir: Optional[str] = None,
        config: Optional[Union[str, Dict]] = None,
        task: str = "classification",
    ) -> None:
        super().__init__(work_dir=work_dir)
        self.task = task
        self.config = self.initial_config(config)
        self.config.default_root_dir = self.work_dir
        self.registry = AnomalibRegistry()

    def initial_config(self, config: Optional[Union[str, Dict]] = None):
        if isinstance(config, str) and config.endswith(".yaml"):
            config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        elif config is None:
            config = {}
        return DictConfig(config)

    def train(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        train_dataloader: Union[DataLoader, LightningDataModule],
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[List[torch.optim.Optimizer]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
        distributed: Optional[bool] = None,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        precision: Optional[_PRECISION_INPUT] = None,
        val_interval: Optional[int] = None,
        metrics: Optional[Dict] = None,
        **kwargs,  # Trainer.__init__ arguments
    ):
        super().train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            checkpoint,
            max_iters,
            max_epochs,
            distributed,
            seed,
            deterministic,
            precision,
            val_interval,
        )
        # FIXME: Modify to work with the config file fill + kwargs
        trainer_config = self.config.get("trainer", {})
        for key, value in kwargs.items():
            trainer_config[key] = value

        # configs.distributed = distributed
        if precision is not None:
            trainer_config["precision"] = precision
        if max_epochs is not None:
            trainer_config["max_epochs"] = max_epochs
            trainer_config["max_steps"] = -1
        elif max_iters is not None:
            trainer_config["max_epochs"] = None
            trainer_config["max_steps"] = max_iters
        if deterministic is not None:
            trainer_config["deterministic"] = deterministic
        if val_interval is not None:
            # Validation Interval in Trainer -> val_check_interval
            trainer_config["val_check_interval"] = val_interval

        # TODO: Need to re-check
        datamodule = trainer_config.pop("datamodule", None)

        # TODO: Need to check callbacks
        if metrics is None:
            metrics = self.config.get("metrics", {})
        metric_threshold = metrics.get("threshold", {})
        callbacks = [
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

        trainer = Trainer(
            callbacks=callbacks,
            **trainer_config,
        )
        self.config["trainer"] = trainer_config
        if optimizer is not None:
            trainer.optimizers = optimizer

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            datamodule=datamodule,
            ckpt_path=checkpoint,
        )
        results = {"model": model, "checkpoint": trainer.ckpt_path}
        return results

    def validate(
        self,
        model: Optional[Union[torch.nn.Module, pl.LightningModule]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: _PRECISION_INPUT = 32,
        **kwargs,
    ) -> Dict[str, float]:  # Metric (data_class or dict)
        super().validate(
            model,
            val_dataloader,
            checkpoint,
            precision,
        )
        raise NotImplementedError()

    def test(
        self,
        model: Optional[Union[torch.nn.Module, pl.LightningModule]] = None,
        test_dataloader: Optional[DataLoader] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: _PRECISION_INPUT = 32,
        **kwargs,
    ) -> Dict[str, float]:  # Metric (data_class or dict)
        super().test(
            model,
            test_dataloader,
            checkpoint,
            precision,
        )
        raise NotImplementedError()

    def predict(
        self,
        model: Optional[Union[torch.nn.Module, pl.LightningModule]] = None,
        img: Optional[Union[str, np.ndarray, list]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        pipeline: Optional[List[Dict]] = None,
        **kwargs,
    ) -> List[Dict]:
        super().predict(
            model,
            img,
            checkpoint,
            pipeline,
        )
        raise NotImplementedError()

    def export(
        self,
        model: Optional[
            Union[torch.nn.Module, pl.LightningModule]
        ] = None,  # Module with _config OR Model Config OR config-file
        checkpoint: Optional[Union[str, Path]] = None,
        precision: _PRECISION_INPUT = 32,
        task: Optional[str] = None,
        codebase: Optional[str] = None,
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        deploy_config: Optional[str] = None,  # File path only?
        dump_features: bool = False,  # TODO
        device: str = "cpu",
        input_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):  # Output: IR Models
        super().export(
            model,
            checkpoint,
            precision,
        )
        raise NotImplementedError()
