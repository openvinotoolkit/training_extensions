# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX engine components."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

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


class Engine:
    """OTX Engine.

    This class defines the common interface for OTX, including methods for training and testing.

    Example:
    >>> engine = Engine(
        work_dir="output/folder/path",
    )
    """

    def __init__(
        self,
        *,
        work_dir: str | Path | None = None,
        max_epochs: int | None = None,
        seed: int | None = None,
        deterministic: bool | None = False,
        precision: _PRECISION_INPUT | None = 32,
        val_check_interval: int | float | None = 1,
        callbacks: list[Callback] | Callback | None = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        accelerator: str | Accelerator = "auto",
        devices: list[int] | str | int = 1,
        **kwargs,
    ):
        """Initializes the Engine object.

        Args:
            work_dir (str | Path | None, optional): The working directory. Defaults to None.
            max_epochs (int | None, optional): The maximum number of epochs. Defaults to None.
            seed (int | None, optional): The random seed. Defaults to None.
            deterministic (bool | None, optional): Whether to enable deterministic behavior. Defaults to False.
            precision (_PRECISION_INPUT | None, optional): The precision of the model. Defaults to 32.
            val_check_interval (int | float | None, optional): The validation check interval. Defaults to 1.
            callbacks (list[Callback] | Callback | None, optional): The callbacks to be used during training.
            logger (Logger | Iterable[Logger] | bool | None, optional): The logger(s) to be used. Defaults to None.
            accelerator (str | Accelerator, optional): The accelerator to be used. Defaults to "auto".
            devices (list[int] | str | int, optional): The devices to be used. Defaults to "auto".
            **kwargs: Additional keyword arguments for pl.Trainer.
        """
        self.work_dir = work_dir
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
    def from_config(cls, config: EngineConfig | str | Path) -> Engine:
        """Create an instance of the Engine class from a configuration file or object.

        Args:
            config (EngineConfig | str | Path): The path to the configuration file or the configuration object.

        Returns:
            Engine: An instance of the Engine class.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            TypeError: If the configuration object is not of type EngineConfig.

        Example:
        >>> runner = Engine.from_config(
            config="config.yaml",
        )
        """
        engine_args, kwargs = {}, {}
        if isinstance(config, (str, Path)):
            with Path(config).open() as f:
                config_dict = yaml.safe_load(f)["engine"]
            field_names = [f.name for f in fields(EngineConfig)]
            for k in config_dict:
                if k in field_names:
                    engine_args[k] = config_dict[k]
                else:
                    kwargs[k] = config_dict[k]
            engine_cfg = EngineConfig(**engine_args)
        else:
            engine_cfg = config

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
        checkpoint: str | Path | None = None,
    ) -> dict[str, Any]:
        """Trains the model using the provided LightningModule and OTXDataModule.

        Args:
            model (LightningModule): The LightningModule to be trained.
            datamodule (OTXDataModule): The OTXDataModule containing the training data.
            checkpoint (str | Path | None, optional): The path to a checkpoint file to resume training from.

        Returns:
            dict[str, Any]: A dictionary containing the callback metrics from the trainer.

        Example:
        >>> engine.train(
            model=LightningModule(),
            datamodule=OTXDataModule(),
            checkpoint="checkpoint.ckpt",
        )

        CLI Usage:
            1. you can pick a model, and you can run through the dataset.
                ```python
                otx train --model <CONFIG | CLASS_PATH_OR_NAME> --data.config.data_root <DATASET_PATH>
                ```
            2. Of course, you can override the various values with commands.
                ```python
                otx train
                    --model <CONFIG | CLASS_PATH_OR_NAME> --data <CONFIG | CLASS_PATH_OR_NAME>
                    --engine.max_epochs 3
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                otx train --config <config_file_path>
                ```
        """
        self.trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(checkpoint) if checkpoint is not None else checkpoint,
        )

        return self.trainer.callback_metrics

    def test(
        self,
        model: LightningModule | None = None,
        datamodule: OTXDataModule | None = None,
        checkpoint: str | Path | None = None,
    ) -> dict:
        """Run the testing phase of the engine.

        Args:
            model (LightningModule | None, optional): The model to be tested.
            datamodule (OTXDataModule | None, optional): The data module containing the test data.
            checkpoint (str | Path | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.

        Returns:
            dict: Dictionary containing the callback metrics from the trainer.

        Example:
        >>> engine.test(
            model=LightningModule(),
            datamodule=OTXDataModule(),
            checkpoint="checkpoint.ckpt",
        )

        CLI Usage:
            1. you can pick a model.
                ```python
                otx test --model <CONFIG | CLASS_PATH_OR_NAME> --data.config.data_root <DATASET_PATH>
                ```
            2. Of course, you can override the various values with commands.
                ```python
                otx test --model <CONFIG | CLASS_PATH_OR_NAME> --data <CONFIG | CLASS_PATH_OR_NAME>
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                otx test --config <config_file_path>
                ```
        """
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
