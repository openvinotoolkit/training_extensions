"""Module for OTX engine components."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import yaml
from lightning import LightningModule, Trainer, seed_everything

from otx.core.config.device import DeviceConfig
from otx.core.config.engine import EngineConfig
from otx.core.data.module import OTXDataModule
from otx.core.model.module.base import OTXLitModule
from otx.core.types.device import OTXDeviceType
from otx.core.types.task import OTXTaskType
from otx.core.utils.auto_configuration import AutoConfigurator
from otx.core.utils.cache import TrainerArgumentsCache
from otx.core.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_model,
)

if TYPE_CHECKING:
    from lightning import Callback
    from lightning.pytorch.loggers import Logger
    from omegaconf import DictConfig
    from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT

    from otx.core.types.transformer_libs import TransformLibType


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
        task: OTXTaskType | None = None,
        work_dir: str | Path = "./otx-workspace",
        data_root: str | Path | None = None,
        device: OTXDeviceType = OTXDeviceType.auto,
        **kwargs,
    ):
        """Initializes the Engine object.

        Args:
            task (OTXTaskType | None, optional): The Task type want to use in Engine.
            work_dir (str | Path | None, optional): The working directory. Defaults to None.
            device (OTXDeviceType, optional):  The devices to be used. Defaults to "auto".
            **kwargs: Additional keyword arguments for pl.Trainer configuration.
        """
        self.task = task
        self.work_dir = work_dir
        self.device = DeviceConfig(accelerator=device)
        self._cache = TrainerArgumentsCache(
            default_root_dir=self.work_dir,
            accelerator=self.device.accelerator,
            devices=self.device.devices,
            **kwargs,
        )
        self._auto_configurator = AutoConfigurator(
            data_root=data_root,
            task=task,
        )

        self._trainer: Trainer | None = None
        self._model: LightningModule | None = None
        self._datamodule: OTXDataModule | None = None

    """
    General OTX Entry Points
    """

    def train(
        self,
        model: LightningModule | None = None,
        datamodule: OTXDataModule | None = None,
        checkpoint: str | Path | None = None,
        max_epochs: int = 10,
        seed: int | None = None,
        deterministic: bool = False,
        precision: _PRECISION_INPUT | None = "32",
        val_check_interval: int | float | None = 1,
        callbacks: list[Callback] | Callback | None = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Trains the model using the provided LightningModule and OTXDataModule.

        Args:
            model (LightningModule): The LightningModule to be trained.
            datamodule (OTXDataModule): The OTXDataModule containing the training data.
            checkpoint (str | Path | None, optional): The path to a checkpoint file to resume training from.
            max_epochs (int | None, optional): The maximum number of epochs. Defaults to None.
            seed (int | None, optional): The random seed. Defaults to None.
            deterministic (bool | None, optional): Whether to enable deterministic behavior. Defaults to False.
            precision (_PRECISION_INPUT | None, optional): The precision of the model. Defaults to 32.
            val_check_interval (int | float | None, optional): The validation check interval. Defaults to 1.
            callbacks (list[Callback] | Callback | None, optional): The callbacks to be used during training.
            logger (Logger | Iterable[Logger] | bool | None, optional): The logger(s) to be used. Defaults to None.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            dict[str, Any]: A dictionary containing the callback metrics from the trainer.

        Example:
        >>> engine.train(
            model=LightningModule(),
            datamodule=OTXDataModule(),
            checkpoint="checkpoint.ckpt",
            max_epochs=3,
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
                    --engine.max_epochs <EPOCHS, int> --checkpoint <CKPT_PATH, str>
                ```
            3. If you have a ready configuration file, run it like this.
                ```python
                otx train --config <CONFIG_PATH, str>
                ```
        """
        if datamodule is None:
            datamodule = self.datamodule
        if model is None:
            model = self.model

        if seed is not None:
            seed_everything(seed, workers=True)

        self._build_trainer(
            logger=logger,
            callbacks=callbacks,
            precision=precision,
            max_epochs=max_epochs,
            deterministic=deterministic,
            val_check_interval=val_check_interval,
            **kwargs,
        )

        self.trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(checkpoint) if checkpoint is not None else checkpoint,
        )
        self.model = model
        self.datamodule = datamodule

        return self.trainer.callback_metrics

    def test(
        self,
        model: LightningModule | None = None,
        datamodule: OTXDataModule | None = None,
        checkpoint: str | Path | None = None,
        **kwargs,
    ) -> dict:
        """Run the testing phase of the engine.

        Args:
            model (LightningModule | None, optional): The model to be tested.
            datamodule (OTXDataModule | None, optional): The data module containing the test data.
            checkpoint (str | Path | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

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
                otx test
                    --model <CONFIG | CLASS_PATH_OR_NAME> --data.config.data_root <DATASET_PATH, str>
                    --checkpoint <CKPT_PATH, str>
                ```
            2. If you have a ready configuration file, run it like this.
                ```python
                otx test --config <CONFIG_PATH, str> --checkpoint <CKPT_PATH, str>
                ```
        """
        if model is None:
            model = self.model
        if datamodule is None:
            datamodule = self.datamodule

        self._build_trainer(**kwargs)

        self.trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(checkpoint) if checkpoint is not None else checkpoint,
        )

        self.model = model
        self.datamodule = datamodule

        return self.trainer.callback_metrics

    def predict(
        self,
        model: LightningModule | None = None,
        datamodule: OTXDataModule | None = None,
        checkpoint: str | Path | None = None,
        return_predictions: bool | None = None,
        **kwargs,
    ) -> list | None:
        """Run predictions using the specified model and data.

        Args:
            model (LightningModule | None): The model to use for predictions.
            datamodule (OTXDataModule | None): The data module to use for predictions.
            checkpoint (str | Path | None): The path to the checkpoint file to load the model from.
            return_predictions (bool | None): Whether to return the predictions or not.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            list | None: The predictions if `return_predictions` is True, otherwise None.

        Example:
        >>> engine.predict(
            model=LightningModule(),
            datamodule=OTXDataModule(),
            checkpoint="checkpoint.ckpt",
            return_predictions=True,
        )

        CLI Usage:
            1. you can pick a model.
                ```python
                otx predict
                    --model <CONFIG | CLASS_PATH_OR_NAME> --data.config.data_root <DATASET_PATH, str>
                    --checkpoint <CKPT_PATH, str>
                ```
            2. If you have a ready configuration file, run it like this.
                ```python
                otx predict --config <CONFIG_PATH, str> --checkpoint <CKPT_PATH, str>
                ```
        """
        if model is None:
            model = self.model
        if datamodule is None:
            datamodule = self.datamodule

        self._build_trainer(**kwargs)

        self.model = model
        self.datamodule = datamodule

        return self.trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(checkpoint) if checkpoint is not None else checkpoint,
            return_predictions=return_predictions,
        )

    def export(self, *args, **kwargs) -> None:
        """Export the trained model to OpenVINO Intermediate Representation (IR) or ONNX formats."""
        raise NotImplementedError

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
        model = None
        datamodule = None
        if isinstance(config, (str, Path)):
            with Path(config).open() as f:
                config_dict = yaml.safe_load(f)
            data_config = config_dict.pop("data", {})
            datamodule = OTXDataModule.from_config(config=data_config)
            model = instantiate_model(config_dict.pop("model", None))
            engine_args = config_dict.pop("engine", config_dict)
            engine_config = EngineConfig(**engine_args)
        else:
            engine_config = config

        callbacks = instantiate_callbacks(config_dict.pop("callbacks", []))
        logger = instantiate_loggers(config_dict.pop("logger", None))
        engine = cls(
            task=engine_config.task,
            work_dir=engine_config.work_dir,
            callbacks=callbacks,
            logger=logger,
            **config_dict,
        )
        engine._model = model  # noqa: SLF001
        engine._datamodule = datamodule  # noqa: SLF001
        return engine

    def list_models(self) -> list[str]:
        """Returns a list of models associated with the given task.

        Returns:
            list[str]: A list of model names.
        """
        model_dict = self._auto_configurator.load_model_configs()
        return list(model_dict.keys())

    """
    Property and setter functions provided by Engine.
    """

    @property
    def trainer(self) -> Trainer:
        """Returns the trainer object associated with the engine.

        To get this property, you should execute `Engine.train()` function first.

        Returns:
            Trainer: The trainer object.
        """
        if self._trainer is None:
            msg = "Please run train() first"
            raise RuntimeError(msg)
        return self._trainer

    def _build_trainer(self, **kwargs) -> None:
        """Instantiate the trainer based on the model parameters."""
        if self._cache.requires_update(**kwargs) or self._trainer is None:
            self._cache.update(**kwargs)
            self._trainer = Trainer(**self._cache.args)
            self.work_dir = self._trainer.default_root_dir

    @property
    def trainer_params(self) -> dict:
        """Returns the parameters used for training the model.

        Returns:
            dict: A dictionary containing the training parameters.
        """
        return self._cache.args

    @property
    def model(self) -> LightningModule:
        """Returns the model object associated with the engine.

        To get this property, you should execute `Engine.train()` function first.

        Returns:
            LightningModule: The LightningModule object.
        """
        if self._model is None:
            self._model = self.set_otx_model()
        return self._model

    @model.setter
    def model(self, model: LightningModule | str) -> None:
        """Sets the model for the engine.

        Args:
            model (LightningModule): The model to be set.

        Returns:
            None
        """
        if isinstance(model, str):
            model = self.set_otx_model(model)
        self._model = model

    @property
    def datamodule(self) -> OTXDataModule:
        """Returns the OTXDataModule object associated with the engine.

        Returns:
            OTXDataModule: The OTXDataModule object.
        """
        if self._datamodule is None:
            self._datamodule = self.set_datamodule()
        return self._datamodule

    @datamodule.setter
    def datamodule(self, datamodule: OTXDataModule) -> None:
        """Sets the datamodule for the engine.

        Args:
            datamodule (OTXDataModule): The datamodule to be set.

        Returns:
            None
        """
        self._datamodule = datamodule

    @property
    def data_root(self) -> str | Path:
        """Returns the data root path.

        Returns:
            str | Path: The data root path.
        """
        return self._auto_configurator.data_root

    @data_root.setter
    def data_root(self, data_root: str | Path) -> None:
        """Set the data root directory for the engine.

        Args:
            data_root (str | Path): The path to the data root directory.

        Returns:
            None
        """
        self._auto_configurator.data_root = data_root

    """
    Functions and properties for Advanced-API users.
    """

    @property
    def default_otx_model_config(self) -> dict | DictConfig:
        """Returns the default model configuration from AutoConfigurator.

        Returns:
            dict | DictConfig: The default model configuration from _auto_configurator.
        """
        if not self._auto_configurator.model_cfg:
            self._auto_configurator.load_default_model_config()
        return self._auto_configurator.model_cfg

    def set_otx_model(
        self,
        model_name: str | None = None,
        overrides: dict | None = None,
    ) -> OTXLitModule:
        """Retrieves the specified model based on the given model name.

        Args:
            model_name (str): The name of the model to retrieve.

        Returns:
            OTXLitModule: The retrieved model.
        """
        if model_name is not None:
            self._auto_configurator.load_default_model_config(model_name=model_name)
        self._model = OTXLitModule.from_config(self.default_otx_model_config, overrides=overrides)
        return self._model

    def set_datamodule(
        self,
        data_root: str | Path | None = None,
        data_format: str | None = None,
        mem_cache_size: str | None = None,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        include_polygons: bool | None = None,
        batch_size: int | dict[str, int] | None = None,
        num_workers: int | dict[str, int] | None = None,
        transforms: list | dict[str, list] | None = None,
        transform_lib_type: TransformLibType | dict[str, TransformLibType] | None = None,
    ) -> OTXDataModule:
        """Returns an instance of OTXDataModule based on the provided configuration.

        Args:
            data_root (str): The root directory of the data.
            task (OTXTaskType): The type of OTX task.

            data_format (str | None, optional): The format of the data. Defaults to None.
            mem_cache_size (str | None, optional): The size of the memory cache. Defaults to None.
            mem_cache_img_max_size (tuple[int, int] | None, optional): The maximum size of images in the memory cache.
                Defaults to None.
            include_polygons (bool | None, optional): Whether to include polygons. Defaults to None.

            batch_size (int | dict[str, int] | None, optional): The batch size. Defaults to None.
            num_workers (int | dict[str, int] | None, optional): The number of workers. Defaults to None.
            transforms (list | dict[str, list] | None, optional): The list of transforms. Defaults to None.
            transform_lib_type (TransformLibType | dict[str, TransformLibType] | None, optional):
                The type of transform library. Defaults to None.

        Returns:
            OTXDataModule: An instance of OTXDataModule based on the provided configuration.
        """
        # Load default DataModuleConfig
        if data_root is None:
            data_root = self._auto_configurator.data_root
        default_config = self._auto_configurator.load_default_data_config()

        # Update DataModuleConfig
        default_config["config"]["data_root"] = data_root
        datamodule_arguments = {
            "data_format": data_format,
            "mem_cache_size": mem_cache_size,
            "mem_cache_img_max_size": mem_cache_img_max_size,
            "include_polygons": include_polygons,
        }
        for datamodule_key, datamodule_value in datamodule_arguments.items():
            if datamodule_value is not None:
                default_config["config"][datamodule_key] = datamodule_value

        # Update SubsetConfig
        subset_arguments = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "transforms": transforms,
            "transform_lib_type": transform_lib_type,
        }
        for subset_key, subset_value in subset_arguments.items():
            if subset_value is not None:
                if isinstance(subset_value, dict):
                    for subset_name, value in subset_value.items():
                        default_config["config"][subset_name][subset_key] = value
                else:
                    default_config["config"]["train_subset"][subset_key] = subset_value
                    default_config["config"]["val_subset"][subset_key] = subset_value
                    default_config["config"]["test_subset"][subset_key] = subset_value

        # Return OTXDataModule
        self._datamodule = OTXDataModule.from_config(config=default_config)
        return self._datamodule
