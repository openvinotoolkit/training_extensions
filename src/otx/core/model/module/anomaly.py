"""Anomaly Lightning OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING, Any, Sequence

import torch
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization.min_max_normalization import _MinMaxNormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

from otx.core.data.entity.base import OTXBatchDataEntity
from otx.core.model.entity.anomaly import OTXAnomalyModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from anomalib.models import AnomalyModule


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
        module = importlib.import_module(f"anomalib.models.image.{self.model}.lightning_model")
        model_class = getattr(module, f"{self.name.title()}")
        self.anomaly_lightning_model = model_class(**self.model.anomalib_lightning_args.get())
        # assign OTXModel's torch model to AnomalyModule's torch model
        self.anomaly_lightning_model.model = self.model.model

    def training_step(self, inputs: OTXBatchDataEntity, batch_idx: int) -> torch.Tensor:
        """Route training step to anomalib's lightning model's training step."""
        inputs = self.model._customize_inputs(inputs)  # noqa: SLF001
        # inputs = self._customize_inputs(inputs)
        return self.anomaly_lightning_model.training_step(inputs, batch_idx)

    def validation_step(self, inputs: ..., **kwargs: ...) -> ...:
        """Route validation step to anomalib's lightning model's validation step."""
        inputs = self.model._customize_inputs(inputs)  # noqa: SLF001
        return self.anomaly_lightning_model.validation_step(inputs, **kwargs)

    def test_step(self, inputs: ..., **kwargs: ...) -> ...:
        """Route test step to anomalib's lightning model's test step."""
        inputs = self.model._customize_inputs(inputs)  # noqa: SLF001
        return self.anomaly_lightning_model.test_step(inputs, **kwargs)

    def predict_step(self, inputs: ..., **kwargs: ...) -> ...:
        """Route predict step to anomalib's lightning model's predict step."""
        inputs = self.model._customize_inputs(inputs)  # noqa: SLF001
        return self.anomaly_lightning_model.predict_step(inputs, **kwargs)

    def configure_optimizers(self) -> dict[str, Any]:
        # TODO
        if self.anomaly_lightning_model.configure_optimizers():
            return {"optimizer": self.optimizer}

    def configure_callbacks(self) -> Sequence[Callback]:
        """Get all necessary callbacks required for training and post-processing on Anomalib models."""
        return [
            _PostProcessorCallback(),
            _MinMaxNormalizationCallback(),
            _ThresholdCallback(threshold="F1AdaptiveThreshold"),
            _MetricsCallback(image_metrics="AUPRO"),  # TODO
        ]

    def forward(self, inputs: T_OTXBatchDataEntity) -> T_OTXBatchPredEntity:
        inputs: torch.Tensor = self.model._customize_inputs()
        outputs = self.anomaly_lightning_model.forward(inputs)
        return self.model._customize_outputs(outputs=outputs, inputs=inputs)
