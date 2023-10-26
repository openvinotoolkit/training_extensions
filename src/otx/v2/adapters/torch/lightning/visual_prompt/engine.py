"""OTX adapters.torch.lightning.anomalib.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader

from otx.v2.adapters.torch.lightning.engine import LightningEngine

from .registry import VisualPromptRegistry

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from pytorch_lightning.utilities.types import EVAL_DATALOADERS

PREDICT_FORMAT = Union[str, Path, np.ndarray]


class VisualPromptEngine(LightningEngine):
    """Anomalib engine using PyTorch and PyTorch Lightning."""

    def __init__(
        self,
        work_dir: str | Path | None = None,
        config: str | dict | None = None,
        task: str = "visual_prompting",
    ) -> None:
        """Initialize the Anomalib engine.

        Args:
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
            config (Optional[Union[str, dict]], optional): The configuration for the engine. Defaults to None.
            task (str, optional): The task to perform. Defaults to "classification".
        """
        super().__init__(work_dir=work_dir, config=config, task=task)
        self.registry = VisualPromptRegistry()

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
        return [CSVLogger(save_dir=self.work_dir, name=target_path, version=self.timestamp)]

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
        if callbacks is not None:
            if isinstance(callbacks, list):
                return callbacks
            if isinstance(callbacks, pl.Callback):
                return [callbacks]
        else:
            callbacks = DictConfig({"checkpoint": {}, "early_stopping": {"monitor": "val_Dice"}})

        callback_list = [
            TQDMProgressBar(),
        ]
        if mode in ("train_val", "train"):
            callback_list.extend([
                ModelCheckpoint(dirpath=self.work_dir, filename="{epoch:02d}", **callbacks.checkpoint),
                LearningRateMonitor(),
            ])
            if mode == "train_val":
                callback_list.append(EarlyStopping(**callbacks.early_stopping))
        return callback_list

    def predict(
        self,
        model: torch.nn.Module | pl.LightningModule | None = None,
        img: PREDICT_FORMAT | (EVAL_DATALOADERS | LightningDataModule) | None = None,
        checkpoint: str | Path | None = None,
        device: list | None = None,  # ["auto", "cpu", "gpu", "cuda"]
        logger: list[Logger] | Logger | bool | None = False,
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
            pass
        elif isinstance(img, (DataLoader, LightningDataModule)):
            dataloader = [img]
        return super().predict(
            model=model,
            img=dataloader,
            checkpoint=checkpoint,
            device=device,
            logger=logger,
            callbacks=callbacks,
        )

    def _onnx_export(
        self,
        model: torch.nn.Module | pl.LightningModule,
        export_path: Path,
        dummy_inputs: dict,
        output_names: list,
        dynamic_axes: dict | None = None,
    ) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with export_path.open("wb") as f:
                torch.onnx.export(
                    model,
                    tuple(dummy_inputs.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )

    def _onnx_to_ir(
        self,
        onnx_model_path: Path,
        export_path: Path,
        command: list,
    ) -> None:

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
        _model = self.latest_model.get("model", None) if model is None else model
        _model = getattr(_model, "model", _model)
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
        if _model is not None and checkpoint is not None:
            state_dict = torch.load(checkpoint)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            _model.load_state_dict(state_dict, strict=False)

        onnx_dir = export_dir / "onnx"
        results: dict = {"outputs": {}}
        # 1) visual_prompting_image_encoder
        dummy_inputs = {"images": torch.randn(1, 3, height, width, dtype=torch.float)}
        output_names = ["image_embeddings"]
        dynamic_axes = None
        model_encoder = _model.image_encoder
        self._onnx_export(
            model=model_encoder,
            export_path=onnx_dir / "sam_encoder.onnx",
            dummy_inputs=dummy_inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        results["outputs"]["onnx"]["encoder"] = str(onnx_dir / "sam_encoder.onnx")

        # 2) SAM without backbone
        embed_dim = _model.prompt_encoder.embed_dim
        embed_size = _model.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }
        dummy_inputs = {
            "image_embeddings": torch.zeros(1, embed_dim, *embed_size, dtype=torch.float),
            "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float),
            "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
            "has_mask_input": torch.tensor([[1]], dtype=torch.float),
        }
        output_names = ["iou_predictions", "low_res_masks"]
        self._onnx_export(
            model=_model,
            export_path=onnx_dir / "sam.onnx",
            dummy_inputs=dummy_inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        results["outputs"]["onnx"]["sam"] = str(onnx_dir / "sam.onnx")

        if export_type.upper() == "OPENVINO":
            ir_dir = export_dir / "openvino"
            ir_dir.mkdir(exist_ok=True, parents=True)
            for onnx_key, model_path in results["outputs"]["onnx"].items():
                optimize_command = [
                    "mo",
                    "--input_model",
                    model_path,
                    "--output_dir",
                    ir_dir,
                    "--model_name",
                    onnx_key,
                ]
                if onnx_key == "encoder":
                    optimize_command += [
                        "--mean_values",
                        str(self.config.dataset.normalize.mean).replace(", ", ","),
                        "--scale_values",
                        str(self.config.dataset.normalize.std).replace(", ", ","),
                    ]
                if precision in ("16", 16, "fp16"):
                    optimize_command.append("--compress_to_fp16")
                _ = run(args=optimize_command, check=False)
                bin_file = Path(ir_dir) / "openvino.bin"
                xml_file = Path(ir_dir) / "openvino.xml"
                if bin_file.exists() and xml_file.exists():
                    results["outputs"][onnx_key]["bin"] = str(Path(ir_dir) / f"{onnx_key}.bin")
                    results["outputs"][onnx_key]["xml"] = str(Path(ir_dir) / f"{onnx_key}.xml")
                else:
                    msg = "OpenVINO Export failed."
                    raise RuntimeError(msg)
        return results
