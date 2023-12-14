# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX engine components."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Union

import yaml
from lightning import LightningModule, Trainer, seed_everything

from otx.core.config.engine import EngineConfig
from otx.core.data.module import OTXDataModule
from otx.core.engine.utils.instantiators import (
        instantiate_callbacks,
        instantiate_loggers,
    )

if TYPE_CHECKING:
    from lightning import Callback
    from lightning.pytorch.accelerators import Accelerator
    from lightning.pytorch.loggers import Logger
    from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT

PATHTYPE = Union[str, Path]


def load_engine_config_from_yaml(file_path: str | Path) -> tuple[EngineConfig, dict]:
    """Load engine configuration from a YAML file.

    Args:
        file_path (str | Path): The path to the YAML file.

    Returns:
        EngineConfig: The engine configuration object.
    """
    from dataclasses import fields
    with Path(file_path).open() as f:
        config_dict = yaml.safe_load(f)["engine"]
    field_names = [f.name for f in fields(EngineConfig)]
    engine_args, trainer_kwargs = {}, {}
    for k in config_dict:
        if k in field_names:
            engine_args[k] = config_dict[k]
        else:
            trainer_kwargs[k] = config_dict[k]
    return EngineConfig(**engine_args), trainer_kwargs


class Engine:
    """OTX Engine class."""

    def __init__(
        self,
        *,
        work_dir: PATHTYPE | None = None,
        max_epochs: int | None = None,
        seed: int | None = None,
        deterministic: bool | None = False,
        precision: _PRECISION_INPUT | None = 32,
        val_check_interval: int | float | None = 1,
        callbacks: list[Callback] | Callback | None = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        accelerator: str | Accelerator = "auto",
        devices: list[int] | str | int = "auto",
        **kwargs,
    ):
        if seed is not None:
            seed_everything(seed, workers=True)

        self._trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            max_epochs=max_epochs,
            deterministic=deterministic,
            val_check_interval=val_check_interval,
            default_root_dir=work_dir,
            **kwargs,
        )

    @property
    def trainer(self) -> Trainer:
        """Returns the trainer object associated with the engine.

        Returns:
            Trainer: The trainer object.
        """
        return self._trainer

    @classmethod
    def from_config(cls, cfg: EngineConfig | PATHTYPE) -> Engine:
        """Create an instance of the Engine class from a configuration file or object.

        Args:
            cfg (EngineConfig | str | Path): The path to the configuration file or the configuration object.

        Returns:
            Engine: An instance of the Engine class.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            TypeError: If the configuration object is not of type EngineConfig.
        """
        if isinstance(cfg, (str, Path)):
            engine_cfg, kwargs = load_engine_config_from_yaml(cfg)
        callbacks = instantiate_callbacks(engine_cfg.callbacks)
        logger = instantiate_loggers(engine_cfg.logger)
        return cls(
            work_dir=engine_cfg.work_dir,
            max_epochs=engine_cfg.max_epochs,
            seed=engine_cfg.seed,
            deterministic=engine_cfg.deterministic,
            precision=engine_cfg.precision,
            val_check_interval=engine_cfg.val_check_interval,
            callbacks=callbacks,
            logger=logger,
            accelerator=engine_cfg.accelerator,
            devices=engine_cfg.devices,
            **kwargs,
        )

    def train(
        self,
        model: LightningModule,
        datamodule: OTXDataModule,
        checkpoint: PATHTYPE | None = None,
    ) -> dict[str, Any]:
        """Trains the model using the provided LightningModule and OTXDataModule.

        Args:
            model (LightningModule): The LightningModule to be trained.
            datamodule (OTXDataModule): The OTXDataModule containing the training data.
            checkpoint (PATHTYPE | None, optional): The path to a checkpoint file to resume training from.

        Returns:
            dict[str, Any]: A dictionary containing the callback metrics from the trainer.
        """
        self.trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(checkpoint) if checkpoint is not None else checkpoint,
        )

        return self.trainer.callback_metrics

    def test(
        self,
        model: LightningModule,
        datamodule: OTXDataModule,
        checkpoint: PATHTYPE | None = None,
    ) -> dict:
        """Test the model using PyTorch Lightning Trainer."""
        self.trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(checkpoint) if checkpoint is not None else checkpoint,
        )

        return self.trainer.callback_metrics

    def predict(self, *args, **kwargs) -> None:
        """Predict with the trained model."""
        raise NotImplementedError

    def export(self, *args, **kwargs) -> None:
        """Export the trained model to OpenVINO Intermediate Representation (IR) or ONNX formats."""
        raise NotImplementedError
