"""Anomaly Lightning OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Sequence

import torch
from anomalib import TaskType
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization.min_max_normalization import _MinMaxNormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

from otx.core.data.entity.anomaly.classification import AnomalyClassificationDataBatch, AnomalyClassificationPrediction
from otx.core.model.entity.anomaly import OTXAnomalyModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from anomalib.models import AnomalyModule
    from lightning import LightningModule, Trainer
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.utilities.types import STEP_OUTPUT


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


class OTXBaseAnomalyLitModel(OTXLitModule):
    """Anomaly OTX Lightning model.

    Used to wrap all the Anomaly models in OTX.
    """

    def __init__(
        self,
        otx_model: OTXAnomalyModel,
        torch_compile: bool,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
        task_type: TaskType,
    ):
        super().__init__(otx_model=otx_model, torch_compile=torch_compile, optimizer=optimizer, scheduler=scheduler)
        self.anomaly_lightning_model: AnomalyModule
        self.model: OTXAnomalyModel
        self.task_type = task_type
        self._setup_anomalib_lightning_model(name=self.model.__class__.__name__)

    def _setup_anomalib_lightning_model(self, name: str | None = None) -> None:
        """Initializes the Anomalib lightning model."""
        module = importlib.import_module(
            f"anomalib.models.image.{name.lower()}.lightning_model",
        )
        model_class = getattr(module, f"{name.title()}")
        self.anomaly_lightning_model = model_class(**self.model.anomalib_lightning_args.get())

    def setup(self, stage: str) -> None:
        """Assign OTXModel's torch model to AnomalyModule's torch model.

        Also connects a few more methods from the Anomalib model to the OTX model.
        """
        # assign OTXModel's torch model to AnomalyModule's torch model
        self.anomaly_lightning_model.model = self.model.model

        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "config"):
            if hasattr(self.trainer.datamodule.config, "test_subset"):
                self.model.transforms = self.trainer.datamodule.config.test_subset.transforms
            elif hasattr(self.trainer.datamodule.config, "val_subset"):
                self.model.transforms = self.trainer.datamodule.config.val_subset.transforms

        self._set_metrics_in_torch()

        self.anomaly_lightning_model.log = self.log
        self.anomaly_lightning_model.trainer = self.trainer
        return super().setup(stage)

    def training_step(self, inputs: AnomalyClassificationDataBatch, batch_idx: int) -> torch.Tensor:
        """Route training step to anomalib's lightning model's training step."""
        inputs = self._customize_inputs(inputs)
        # inputs = self._customize_inputs(inputs)
        return self.anomaly_lightning_model.training_step(inputs, batch_idx)

    def validation_step(self, inputs: AnomalyClassificationDataBatch, batch_idx: int = 0, **kwargs: Any) -> ...:
        """Route validation step to anomalib's lightning model's validation step."""
        inputs = self._customize_inputs(inputs)
        return self.anomaly_lightning_model.validation_step(inputs, batch_idx, **kwargs)

    def on_validation_end(self) -> None:
        self.anomaly_lightning_model.on_validation_end()
        # assign the updated values to the OTX model
        self._set_metrics_in_torch()

    def _set_metrics_in_torch(self):
        self.model.pixel_threshold = self.anomaly_lightning_model.pixel_threshold.value.cpu().numpy().tolist()
        self.model.image_threshold = self.anomaly_lightning_model.image_threshold.value.cpu().numpy().tolist()
        self.model.min_val = (
            self.anomaly_lightning_model.normalization_metrics.state_dict()["min"].cpu().numpy().tolist()
        )
        self.model.max_val = (
            self.anomaly_lightning_model.normalization_metrics.state_dict()["max"].cpu().numpy().tolist()
        )

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
        image_metrics = ["AUROC", "F1Score"]
        pixel_metrics = image_metrics if self.task_type != TaskType.CLASSIFICATION else None
        return _RouteCallback(
            [
                _PostProcessorCallback(),
                _MinMaxNormalizationCallback(),  # ModelAPI only supports min-max normalization as of now
                _ThresholdCallback(threshold="F1AdaptiveThreshold"),
                _MetricsCallback(
                    task=self.task_type,
                    image_metrics=image_metrics,
                    pixel_metrics=pixel_metrics,
                ),
            ],
        )

    def forward(self, inputs: AnomalyClassificationDataBatch) -> AnomalyClassificationPrediction:
        """Wrap forward method of the Anomalib model."""
        inputs: torch.Tensor = self._customize_inputs()
        outputs = self.anomaly_lightning_model.forward(inputs)
        return self._customize_outputs(outputs=outputs, inputs=inputs)

    def state_dict(self) -> dict[str, Any]:
        """Set keys of state_dict to allow correct loading of the model."""
        state_dict = super().state_dict()
        state_dict["anomaly_lightning_model_class"] = self.anomaly_lightning_model.__class__.__name__

        # remove model.model from state_dict as it is already present in anomaly_lightning_model.model
        # this saves space
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("model.model")}
        # reorder keys
        extra_info_keys = ("image_threshold_class", "pixel_threshold_class", "normalization_class")
        for key in extra_info_keys:
            if key in state_dict:
                state_dict[f"anomaly_lightning_model.{key}"] = state_dict[key]
        for key in extra_info_keys:
            if key in state_dict:
                state_dict.pop(key)

        return state_dict

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state_dict and initialize the Anomalib model.

        Also assigns the keys that were removed when saving the state_dict to save disk space.
        """
        anomaly_lightning_module = ckpt["state_dict"].pop("anomaly_lightning_model_class")
        self._setup_anomalib_lightning_model(name=anomaly_lightning_module)
        # extract anomaly_lightning_model's state_dict from ckpt and load it
        anomaly_lightning_module_keys = [key for key in ckpt["state_dict"] if key.startswith("anomaly_lightning_model")]
        anomaly_lightning_module_state_dict = {}
        for key, value in ckpt["state_dict"].items():
            if key in anomaly_lightning_module_keys:
                anomaly_lightning_module_state_dict[key.split("anomaly_lightning_model.")[1]] = value

        self.anomaly_lightning_model.load_state_dict(anomaly_lightning_module_state_dict)

        # restore keys for model.model
        for key in anomaly_lightning_module_keys:
            if key.startswith("anomaly_lightning_model.model"):
                ckpt["state_dict"][f"model.model.{key.split('anomaly_lightning_model.model.')[1]}"] = ckpt[
                    "state_dict"
                ][key]

        # remove extra info keys
        extra_info_keys = ("image_threshold_class", "pixel_threshold_class", "normalization_class", "_is_fine_tuned")
        for key in extra_info_keys:
            if f"anomaly_lightning_model.{key}" in ckpt["state_dict"]:
                ckpt["state_dict"].pop(f"anomaly_lightning_model.{key}")

        return super().load_state_dict(ckpt, *args, **kwargs)

    def _customize_outputs(
        self,
        outputs: Any,
        inputs: AnomalyClassificationDataBatch,
    ) -> AnomalyClassificationPrediction:
        # TODO
        ...


class OTXAnomalyClassificationLitModel(OTXBaseAnomalyLitModel):
    def __init__(
        self,
        otx_model: OTXAnomalyModel,
        torch_compile: bool,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
    ):
        super().__init__(otx_model, torch_compile, optimizer, scheduler, task_type=TaskType.CLASSIFICATION)

    def _customize_inputs(self, inputs: AnomalyClassificationDataBatch) -> dict[str, Any]:  # anomalib model inputs
        """Customize inputs for the model."""
        return {"image": inputs.images, "label": torch.vstack(inputs.labels).squeeze()}


class OTXAnomalySegmentationLitModel(OTXBaseAnomalyLitModel):
    def __init__(
        self,
        otx_model: OTXAnomalyModel,
        torch_compile: bool,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
    ):
        super().__init__(otx_model, torch_compile, optimizer, scheduler, task_type=TaskType.SEGMENTATION)

    def _customize_inputs(self, inputs: AnomalyClassificationDataBatch) -> dict[str, Any]:  # anomalib model inputs
        """Customize inputs for the model."""
        return {"image": inputs.images, "label": torch.vstack(inputs.labels).squeeze(), "mask": inputs.masks}


class OTXAnomalyDetectionLitModel(OTXBaseAnomalyLitModel):
    def __init__(
        self,
        otx_model: OTXAnomalyModel,
        torch_compile: bool,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
    ):
        super().__init__(otx_model, torch_compile, optimizer, scheduler, task_type=TaskType.DETECTION)

    def _customize_inputs(self, inputs: AnomalyClassificationDataBatch) -> dict[str, Any]:  # anomalib model inputs
        """Customize inputs for the model."""
        return {
            "image": inputs.images,
            "label": torch.vstack(inputs.labels).squeeze(),
            "mask": inputs.masks,
            "boxes": inputs.boxes,
        }
