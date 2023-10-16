"""OTX adapters.torch.anomalib.Engine API."""

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
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.loggers import AnomalibTensorBoardLogger
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader

from otx.v2.api.core.engine import Engine
from otx.v2.api.utils import set_tuple_constructor

from .registry import AnomalibRegistry

if TYPE_CHECKING:
    from pytorch_lightning.trainer.connectors.accelerator_connector import (
        _PRECISION_INPUT,
    )
    from pytorch_lightning.utilities.types import EVAL_DATALOADERS
    from torch.optim import Optimizer

PREDICT_FORMAT = Union[str, Path, np.ndarray]


class AnomalibEngine(Engine):
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
        super().__init__(work_dir=work_dir)
        self.trainer: Trainer
        self.trainer_config: dict = {}
        self.latest_model = {"model": None, "checkpoint": None}
        self.task = task
        self.config = self._initial_config(config)
        if hasattr(self, "work_dir"):
            self.config.default_root_dir = self.work_dir
        self.registry = AnomalibRegistry()

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
        if deterministic is not None:
            self.trainer_config["deterministic"] = deterministic
        if val_interval is not None:
            # Validation Interval in Trainer -> val_check_interval
            self.trainer_config["val_check_interval"] = val_interval

        return update_check

    def get_callbacks(self, metrics: dict | None = None) -> list:
        """Return a list of callbacks to be used during training.

        Args:
            metrics (Optional[dict]): A dictionary containing the metrics to be used during training.

        Returns:
            list: A list of callbacks to be used during training.
        """
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
            **kwargs: Additional arguments to pass to the Trainer constructor.

        Returns:
            dict: A dictionary containing the trained model and the path to the saved checkpoint.
        """
        _ = distributed
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
        metrics = self.trainer_config.pop("metrics", None)
        logger = self.trainer_config.pop("logger", True)
        num_sanity_val_steps = 0
        if val_dataloader is not None:
            num_sanity_val_steps = self.trainer_config.pop("num_sanity_val_steps", num_sanity_val_steps)

        target_folder = f"{self.timestamp}_train"
        if logger is True or not logger:
            logger = [AnomalibTensorBoardLogger(save_dir=self.work_dir, name=target_folder)]
        else:
            pass

        if not hasattr(self, "trainer") or update_check:
            callbacks = self.get_callbacks(metrics=metrics)
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                num_sanity_val_steps=num_sanity_val_steps,
                **self.trainer_config,
            )
        self.config["trainer"] = self.trainer_config
        if optimizer is not None:
            self.trainer.optimizers = optimizer

        self.trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            datamodule=datamodule,
            ckpt_path=checkpoint,
        )

        output_model_dir = self.work_dir / target_folder / "models"
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
            **kwargs: Additional keyword arguments to pass to the method.

        Returns:
            dict: The validation metric (data_class or dict).
        """
        update_check = self._update_config(func_args={"precision": precision}, **kwargs)

        datamodule = self.trainer_config.pop("datamodule", None)
        metrics = self.trainer_config.pop("metrics", None)
        logger = self.trainer_config.pop("logger", True)
        target_folder = f"{self.timestamp}_validate"
        if logger is True or not logger:
            logger = [AnomalibTensorBoardLogger(save_dir=self.work_dir, name=target_folder)]

        if not hasattr(self, "trainer") or update_check:
            callbacks = self.get_callbacks(metrics=metrics)
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                **self.trainer_config,
            )
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint")
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
            **kwargs: Additional keyword arguments to pass to the method.

        Returns:
            dict: The test results as a dictionary.
        """
        _ = self._update_config(func_args={"precision": precision}, **kwargs)
        if model is None:
            model = self.latest_model.get("model", None)
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)

        logger = self.trainer_config.pop("logger", True)
        target_folder = f"{self.timestamp}_test"
        if logger is True or not logger:
            logger = [AnomalibTensorBoardLogger(save_dir=self.work_dir, name=target_folder)]
        callbacks = self.get_callbacks()
        self.trainer = Trainer(
            logger=logger,
            callbacks=callbacks,
            resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None,
            **self.trainer_config,
        )
        return self.trainer.test(
            model=model,
            dataloaders=[test_dataloader],
        )

    def predict(
        self,
        model: torch.nn.Module | pl.LightningModule | None = None,
        img: PREDICT_FORMAT | (EVAL_DATALOADERS | LightningDataModule) | None = None,
        checkpoint: str | Path | None = None,
        device: list | None = None,  # ["auto", "cpu", "gpu", "cuda"]
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
        if model is None:
            model = self.latest_model.get("model", None)
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)
        if device is None:
            device = self.trainer_config.pop("device", None)

        logger = self.trainer_config.pop("logger", True)
        target_folder = f"{self.timestamp}_predict"
        if logger is True or not logger:
            logger = [AnomalibTensorBoardLogger(save_dir=self.work_dir, name=target_folder)]

        callbacks = self.get_callbacks()
        trainer = Trainer(
            logger=logger,
            callbacks=callbacks,
            resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None,
            **self.trainer_config,
        )

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
            dataset = InferenceDataset(img, image_size=image_size, transform=transform)
            dataloader = DataLoader(dataset)
        elif isinstance(img, (DataLoader, LightningDataModule)):
            dataloader = [img]
        # Lightning Inferencer
        return trainer.predict(
            model=model,
            dataloaders=[dataloader],
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
            device (Optional[str]): The device to use for exporting the model.
            input_shape (Optional[Tuple[int, int]]): The input shape to use for exporting the model.

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
        if device is None:
            device = getattr(model, "device", None)

        export_dir = self.work_dir / f"{self.timestamp}_export"
        export_dir.mkdir(exist_ok=True, parents=True)

        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)
        # if "model" in state_dict:
        # if "state_dict" in state_dict:

        # Torch to onnx
        onnx_dir = export_dir / "onnx"
        onnx_dir.mkdir(exist_ok=True, parents=True)
        onnx_model = str(onnx_dir / "onnx_model.onnx")
        torch.onnx.export(
            model=getattr(_model, "model", _model),
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
