"""OTX adapters.torch.anomalib.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import subprocess  # nosec B404
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _PRECISION_INPUT,
)
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from otx.v2.api.core.engine import Engine

from .registry import AnomalibRegistry

PREDICT_FORMAT = Union[str, Path, np.ndarray]


class AnomalibEngine(Engine):
    def __init__(
        self,
        work_dir: Optional[Union[str, Path]] = None,
        config: Optional[Union[str, Dict]] = None,
        task: str = "classification",
    ) -> None:
        super().__init__(work_dir=work_dir)
        self.trainer: Trainer
        self.trainer_config: Dict = {}
        self.latest_model = {"model": None, "checkpoint": None}
        self.task = task
        self.config = self._initial_config(config)
        self.config.default_root_dir = self.work_dir
        self.registry = AnomalibRegistry()

    def _initial_config(self, config: Optional[Union[str, Dict]] = None):
        if isinstance(config, str) and config.endswith(".yaml"):
            config = yaml.load(open(config), Loader=yaml.FullLoader)
        elif config is None:
            config = {}
        return DictConfig(config)

    def _update_config(
        self,
        func_args: Dict,
        **kwargs,
    ):
        update_check = not all(value is None for value in func_args.values()) or not all(
            value is None for value in kwargs.values()
        )
        # FIXME: Modify to work with the config file fill + kwargs
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

    def get_callbacks(self, metrics: Optional[Dict] = None):
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
        return callbacks

    def train(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        train_dataloader: Union[DataLoader, LightningDataModule],
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Union[dict, Optimizer]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
        distributed: Optional[bool] = None,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        precision: Optional[_PRECISION_INPUT] = None,
        val_interval: Optional[int] = None,
        **kwargs,  # Trainer.__init__ arguments
    ):
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

        target_folder = f"{self.timestamp}_train"
        if logger is True or not logger:
            logger = [AnomalibTensorBoardLogger(save_dir=self.work_dir, name=target_folder)]
        else:
            # TODO: How to change custom folder output?
            pass

        if not hasattr(self, "trainer") or update_check:
            callbacks = self.get_callbacks(metrics=metrics)
            self.trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
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
        model: Optional[Union[torch.nn.Module, pl.LightningModule]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[_PRECISION_INPUT] = None,
        **kwargs,
    ) -> Dict[str, float]:  # Metric (data_class or dict)
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
        results = self.trainer.validate(
            model=model,
            dataloaders=val_dataloader,
            ckpt_path=checkpoint,
            datamodule=datamodule,
        )
        return results

    def test(
        self,
        model: Optional[Union[torch.nn.Module, pl.LightningModule]] = None,
        test_dataloader: Optional[DataLoader] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[_PRECISION_INPUT] = None,
        **kwargs,
    ) -> Dict[str, float]:  # Metric (data_class or dict)
        _ = self._update_config(func_args={"precision": precision}, **kwargs)
        if model is None:
            model = self.latest_model.get("model", None)
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)

        # TODO: Need to check re-utilize existing Trainer

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
        results = self.trainer.test(
            model=model,
            dataloaders=[test_dataloader],
        )
        return results

    def predict(
        self,
        model: Optional[Union[torch.nn.Module, pl.LightningModule]] = None,
        img: Optional[Union[PREDICT_FORMAT, EVAL_DATALOADERS, LightningDataModule]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        pipeline: Optional[Union[Dict, List]] = None,
        device: str = "auto",  # ["auto", "cpu", "gpu", "cuda"]
        visualization_mode: str = "simple",  # ["full", "simple"]
        **kwargs,
    ) -> List[Dict]:
        results = []
        if model is None:
            model = self.latest_model.get("model", None)
        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)

        logger = self.trainer_config.pop("logger", True)
        target_folder = f"{self.timestamp}_predict"
        if logger is True or not logger:
            logger = [AnomalibTensorBoardLogger(save_dir=self.work_dir, name=target_folder)]

        callbacks = self.get_callbacks()
        # TODO: Need to check re-utilize existing Trainer
        trainer = Trainer(
            logger=logger,
            callbacks=callbacks,
            resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None,
            **self.trainer_config,
        )

        dataloader = None
        if isinstance(img, (str, Path)):
            dataset_config = self.config.get("dataset", {})
            transform_config = (
                dataset_config.transform_config.eval if "transform_config" in dataset_config.keys() else None
            )
            image_size = dataset_config.get("image_size", (256, 256))
            center_crop = dataset_config.get("center_crop")
            if center_crop is not None:
                center_crop = tuple(center_crop)
            normalization = InputNormalizationMethod(dataset_config.get("normalization", "imagenet"))
            transform = get_transforms(
                config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
            )
            dataset = InferenceDataset(img, image_size=image_size, transform=transform)  # type: ignore
            dataloader = DataLoader(dataset)
        elif isinstance(img, (DataLoader, LightningDataModule)):
            dataloader = [img]
        else:
            # TODO: img -> np.ndarray?
            pass
        # Lightning Inferencer
        results = trainer.predict(
            model=model,
            dataloaders=[dataloader],
        )
        return results

    def export(
        self,
        model: Optional[
            Union[torch.nn.Module, pl.LightningModule]
        ] = None,  # Module with _config OR Model Config OR config-file
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[_PRECISION_INPUT] = None,
        task: Optional[str] = None,
        codebase: Optional[str] = None,
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        deploy_config: Optional[str] = None,  # File path only?
        dump_features: bool = False,  # TODO
        device: Optional[str] = None,
        input_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, str]]:  # Output: IR Models
        # Set input_shape (input_size)
        if model is None:
            model = self.latest_model.get("model", None)
        model_config = self.config.get("model", {})
        height, width = model_config.get("input_size", (256, 256))
        if input_shape is not None:
            height, width = input_shape

        # Set device
        if device is None:
            device = getattr(model, "device", None)

        export_dir = self.work_dir / f"{self.timestamp}_export"
        export_dir.mkdir(exist_ok=True, parents=True)

        # Torch to onnx
        onnx_dir = export_dir / "onnx"
        onnx_dir.mkdir(exist_ok=True, parents=True)
        onnx_model = str(onnx_dir / "onnx_model.onnx")
        torch.onnx.export(
            model=getattr(model, "model", model),
            args=torch.zeros((1, 3, height, width)).to(device),
            f=onnx_model,
            opset_version=11,
        )

        results: Dict[str, Dict[str, str]] = {"outputs": {}}
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
            _ = subprocess.run(optimize_command)
            bin_file = Path(ir_dir) / "openvino.bin"
            xml_file = Path(ir_dir) / "openvino.xml"
            if bin_file.exists() and xml_file.exists():
                results["outputs"]["bin"] = str(Path(ir_dir) / "openvino.bin")
                results["outputs"]["xml"] = str(Path(ir_dir) / "openvino.xml")
            else:
                raise RuntimeError("OpenVINO Export failed.")
        return results
