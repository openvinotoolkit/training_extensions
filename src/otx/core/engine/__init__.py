# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Module for OTX engine components."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.core.config import TrainConfig
from otx.core.engine.train import train

if TYPE_CHECKING:
    from lightning import Trainer


class Engine:
    """OTX Engine class."""

    def __init__(self) -> None:
        self._trainer = None

    @property
    def trainer(self) -> Trainer:
        """PyTorch Lightning Trainer.

        To get this property, you should execute `Engine.train()` function first.
        """
        if self._trainer is None:
            msg = "Please run train() first"
            raise RuntimeError(msg)

        return self._trainer

    def train(self, cfg: TrainConfig) -> dict[str, Any]:
        """Train the model using PyTorch Lightning Trainer."""
        trainer, metrics = train(cfg)
        self._trainer = trainer
        return metrics

    def predict(self, *args, **kwargs) -> None:
        """Predict with the trained model."""
        raise NotImplementedError

    def export(self, *args, **kwargs) -> None:
        """Export the trained model to OpenVINO Intermediate Representation (IR) or ONNX formats."""
        # return export(
        #     self.cfg.deploy_cfg,
        #     model,
        #     f"{output_path}/openvino",
        #     half_precision,
        #     onnx_only=export_format == ExportType.ONNX,
        # )
        raise NotImplementedError
