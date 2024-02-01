"""Anomaly Lightning OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Sequence

import torch
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization.min_max_normalization import _MinMaxNormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT

from otx.core.data.entity.anomaly.classification import AnomalyClassificationDataBatch, AnomalyClassificationPrediction
from otx.core.model.entity.anomaly import OTXAnomalyModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from anomalib.models import AnomalyModule


class _RouteCallback(Callback):
    def __init__(self, callbacks: Sequence[Callback]):
        self.callbacks = callbacks

    def _call_on_anomalib_model(
        self,
        hook_name: str,
        pl_module: OTXAnomalyLitModel,
        **kwargs: Any,
    ) -> None:
        anomalib_module = pl_module.anomaly_lightning_model
        for callback in self.callbacks:
            if hasattr(callback, hook_name):
                getattr(callback, hook_name)(pl_module=anomalib_module, **kwargs)

    def setup(self, trainer: Trainer, pl_module: OTXAnomalyLitModel, stage: str) -> None:
        self._call_on_anomalib_model(hook_name="setup", pl_module=pl_module, trainer=trainer, stage=stage)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._call_on_anomalib_model(hook_name="on_validation_epoch_start", pl_module=pl_module, trainer=trainer)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._call_on_anomalib_model(
            hook_name="on_validation_batch_end",
            pl_module=pl_module,
            trainer=trainer,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._call_on_anomalib_model(hook_name="on_validation_epoch_end", pl_module=pl_module, trainer=trainer)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._call_on_anomalib_model(hook_name="on_test_start", pl_module=pl_module, trainer=trainer)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._call_on_anomalib_model(hook_name="on_test_epoch_start", pl_module=pl_module, trainer=trainer)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._call_on_anomalib_model(
            hook_name="on_test_batch_end",
            pl_module=pl_module,
            trainer=trainer,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._call_on_anomalib_model(hook_name="on_test_epoch_end", pl_module=pl_module, trainer=trainer)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._call_on_anomalib_model(
            hook_name="on_predict_batch_end",
            pl_module=pl_module,
            trainer=trainer,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )


class OTXAnomalyLitModel(OTXLitModule):
    """Anomaly OTX Lightning model.

    Used to wrap all the Anomaly models in OTX.
    """

    def __init__(
        self,
        otx_model: OTXAnomalyModel,
        torch_compile: bool,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
    ):
        super().__init__(otx_model=otx_model, torch_compile=torch_compile, optimizer=optimizer, scheduler=scheduler)
        self.anomaly_lightning_model: AnomalyModule
        self.model: OTXAnomalyModel
        self._setup_anomalib_lightning_model()

    def _setup_anomalib_lightning_model(self) -> None:
        model_name = self.model.__class__.__name__
        module = importlib.import_module(
            f"anomalib.models.image.{model_name.lower()}.lightning_model",
        )
        model_class = getattr(module, f"{model_name.title()}")
        self.anomaly_lightning_model = model_class(**self.model.anomalib_lightning_args.get())
        # assign OTXModel's torch model to AnomalyModule's torch model
        self.anomaly_lightning_model.model = self.model.model
        self.anomaly_lightning_model.log = self.log

    def training_step(self, inputs: AnomalyClassificationDataBatch, batch_idx: int) -> torch.Tensor:
        """Route training step to anomalib's lightning model's training step."""
        inputs = self._customize_inputs(inputs)
        # inputs = self._customize_inputs(inputs)
        return self.anomaly_lightning_model.training_step(inputs, batch_idx)

    def validation_step(self, inputs: AnomalyClassificationDataBatch, batch_idx: int = 0, **kwargs: Any) -> ...:
        """Route validation step to anomalib's lightning model's validation step."""
        inputs = self._customize_inputs(inputs)
        return self.anomaly_lightning_model.validation_step(inputs, batch_idx, **kwargs)

    def test_step(self, inputs: AnomalyClassificationDataBatch, batch_idx: int = 0, **kwargs: Any) -> ...:
        """Route test step to anomalib's lightning model's test step."""
        inputs = self._customize_inputs(inputs)
        return self.anomaly_lightning_model.test_step(inputs, batch_idx, **kwargs)

    def predict_step(self, inputs: AnomalyClassificationDataBatch, batch_idx: int = 0, **kwargs: Any) -> ...:
        """Route predict step to anomalib's lightning model's predict step."""
        inputs = self._customize_inputs(inputs)
        return self.anomaly_lightning_model.predict_step(inputs, batch_idx, **kwargs)

    def configure_optimizers(self) -> dict[str, Any] | None:
        """Configure optimizers for Anomalib models.

        If the anomalib lightning model supports optimizers, return the optimizer.
        Else don't return optimizer even if it is configured in the OTX model.
        """
        if self.anomaly_lightning_model.configure_optimizers() and self.optimizer:
            return {"optimizer": self.optimizer}
        return None

    def configure_callbacks(self) -> Callback:
        """Get all necessary callbacks required for training and post-processing on Anomalib models."""
        return _RouteCallback(
            [
                _PostProcessorCallback(),
                _MinMaxNormalizationCallback(),
                _ThresholdCallback(threshold="F1AdaptiveThreshold"),
                _MetricsCallback(),
            ],
        )

    def forward(self, inputs: AnomalyClassificationDataBatch) -> AnomalyClassificationPrediction:
        inputs: torch.Tensor = self._customize_inputs()
        outputs = self.anomaly_lightning_model.forward(inputs)
        return self.model._customize_outputs(outputs=outputs, inputs=inputs)

    def _customize_inputs(self, inputs: AnomalyClassificationDataBatch) -> dict[str, Any]:  # anomalib model inputs
        """Customize inputs for the model."""
        return {"image": inputs.images}
