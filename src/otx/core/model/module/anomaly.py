"""Anomaly Lightning OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence

import torch
from anomalib import TaskType
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization.min_max_normalization import _MinMaxNormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from lightning.pytorch.callbacks.callback import Callback

from otx.core.data.dataset.base import LabelInfo
from otx.core.data.entity.anomaly import (
    AnomalyClassificationBatchPrediction,
    AnomalyClassificationDataBatch,
    AnomalyDetectionBatchPrediction,
    AnomalyDetectionDataBatch,
    AnomalySegmentationBatchPrediction,
    AnomalySegmentationDataBatch,
)
from otx.core.data.entity.base import T_OTXBatchDataEntity, T_OTXBatchPredEntity
from otx.core.model.entity.anomaly import OTXAnomalyModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.core.optimizer import LightningOptimizer
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from openvino.model_api.models.anomaly import AnomalyResult
    from torch.optim.optimizer import Optimizer


class _RouteCallback(Callback):
    """Routes the calls with Anomalib's lightning model as the parameter instead of OTXLightning model.

    The callbacks necessary to train the Anomalib model are initialized by the Anomalib's Engine. To ensure
    that the OTXLightning model is trained correctly, the ``configure_optimizer`` of AnomalyOTXLightning model returns
    the necessary callback. Since these callbacks are attached to the OTX Trainer, they are called with OTXLightning
    model as the parameter. This class overrides the parameter with Anomalib's lightning model.
    """

    def __init__(self, callbacks: Sequence[Callback]):
        self.callbacks = callbacks

    def _call_on_anomalib_model(
        self,
        hook_name: str,
        pl_module: LightningModule,
        **kwargs,
    ) -> None:
        anomalib_module = pl_module.model.model
        for callback in self.callbacks:
            if hasattr(callback, hook_name):
                getattr(callback, hook_name)(pl_module=anomalib_module, **kwargs)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self._call_on_anomalib_model(hook_name="setup", pl_module=pl_module, trainer=trainer, stage=stage)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._call_on_anomalib_model(hook_name="on_validation_epoch_start", pl_module=pl_module, trainer=trainer)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch,
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
        outputs: dict,
        batch: AnomalyClassificationDataBatch | AnomalyDetectionDataBatch | AnomalySegmentationDataBatch,
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
        outputs: dict,
        batch: AnomalyClassificationBatchPrediction
        | AnomalySegmentationBatchPrediction
        | AnomalyDetectionBatchPrediction,
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


class OTXBaseAnomalyLitModel(OTXLitModule, ABC, Generic[T_OTXBatchPredEntity, T_OTXBatchDataEntity]):
    """Anomaly OTX Lightning model.

    Used to wrap all the Anomaly models in OTX.
    """

    def __init__(
        self,
        otx_model: OTXAnomalyModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable,
        task_type: TaskType,
    ):
        super().__init__(otx_model=otx_model, torch_compile=torch_compile, optimizer=optimizer, scheduler=scheduler)
        self.model: OTXAnomalyModel
        self.task_type = task_type
        self._meta_info: LabelInfo
        self.model_name = self.model.__class__.__name__
        self._setup_anomalib_lightning_model()

    def _setup_anomalib_lightning_model(self) -> None:
        """Initializes the Anomalib lightning model."""
        if self.model_name == "OVAnomalyModel":
            return  # Ignore loading the lightning model when it is an OpenVINO model.
        self.automatic_optimization = self.model.model.automatic_optimization

    def setup(self, stage: str) -> None:
        """Assign OTXModel's torch model to AnomalyModule's torch model.

        Also connects a few more methods from the Anomalib model to the OTX model.
        """
        if self.model_name == "OVAnomalyModel":
            return None
        self.model.task_type = self.task_type

        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "config"):
            if hasattr(self.trainer.datamodule.config, "test_subset"):
                self.model.extract_model_info_from_transforms(self.trainer.datamodule.config.test_subset.transforms)
            elif hasattr(self.trainer.datamodule.config, "val_subset"):
                self.model.extract_model_info_from_transforms(self.trainer.datamodule.config.val_subset.transforms)

        self._set_metrics_in_torch()

        self.model.model.log = self.log
        self.model.model.trainer = self.trainer
        return super().setup(stage)

    def training_step(self, inputs: T_OTXBatchDataEntity, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        """Route training step to anomalib's lightning model's training step."""
        _inputs = self._customize_inputs(inputs)
        return self.model.model.training_step(_inputs, batch_idx)

    def validation_step(self, inputs: T_OTXBatchDataEntity, batch_idx: int = 0, **kwargs) -> STEP_OUTPUT:
        """Route validation step to anomalib's lightning model's validation step."""
        _inputs = self._customize_inputs(inputs)
        # no need to customize outputs for validation step
        return self.model.model.validation_step(_inputs, batch_idx, **kwargs)

    def on_validation_end(self) -> None:
        """Call anomaly_lightning_model's on_validation_end."""
        self.model.model.on_validation_end()
        # assign the updated values to the OTX model
        self._set_metrics_in_torch()

    def _set_metrics_in_torch(self) -> None:
        """This is needed for OpenVINO export."""
        self.model.model_info.pixel_threshold = self.model.model.pixel_threshold.value.cpu().numpy().tolist()
        self.model.model_info.image_threshold = self.model.model.image_threshold.value.cpu().numpy().tolist()
        min_val = self.model.model.normalization_metrics.state_dict()["min"].cpu().numpy().tolist()
        max_val = self.model.model.normalization_metrics.state_dict()["max"].cpu().numpy().tolist()
        self.model.model_info.normalization_scale = max_val - min_val

    def test_step(
        self,
        inputs: T_OTXBatchDataEntity,
        batch_idx: int = 0,
        **kwargs,
    ) -> T_OTXBatchPredEntity:
        """Route test step to Anomalib's lightning model's test step."""
        if self.model_name == "OVAnomalyModel":
            return self.model(inputs)

        dict_inputs = self._customize_inputs(inputs)
        return self.model.model.test_step(dict_inputs, batch_idx, **kwargs)

    def on_test_batch_end(
        self,
        outputs: dict,
        batch: T_OTXBatchDataEntity,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called in the test loop after the batch.

        Args:
            outputs: The outputs of test_step(x)
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """
        if self.model_name == "OVAnomalyModel":
            return  # Ignore when it is an OpenVINO model.

        # Since outputs need to be replaced inplace, we can't change the datatype of outputs.
        # That's why outputs is cleared and replaced with the new outputs. The problem with this is that
        # Instead of ``engine.test()`` returning [BatchPrediction,...], it returns
        # [{prediction: BatchPrediction}, {...}, ...]
        _outputs = self._customize_outputs(outputs, batch)
        outputs.clear()
        outputs.update({"prediction": _outputs})

    def predict_step(
        self,
        inputs: T_OTXBatchDataEntity,
        batch_idx: int = 0,
        **kwargs,
    ) -> T_OTXBatchPredEntity:
        """Route predict step to anomalib's lightning model's predict step."""
        if self.model_name == "OVAnomalyModel":
            return self.forward(inputs)

        _inputs = self._customize_inputs(inputs)
        return self.model.model.predict_step(_inputs, batch_idx, **kwargs)

    def on_predict_batch_end(
        self,
        outputs: dict,
        batch: T_OTXBatchDataEntity,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called in the predict loop after the batch.

        Args:
            outputs: The outputs of predict_step(x)
            batch: The batched data as it is returned by the prediction DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """
        if self.model_name == "OVAnomalyModel":
            return  # Ignore when it is an OpenVINO model.

        # Since outputs need to be replaced inplace, we can't change the datatype of outputs.
        # That's why outputs is cleared and replaced with the new outputs. The problem with this is that
        # Instead of ``engine.predict()`` returning [BatchPrediction,...], it returns
        # [{prediction: BatchPrediction}, {...}, ...]
        _outputs = self._customize_outputs(outputs, batch)
        outputs.clear()
        outputs.update({"prediction": _outputs})

    def on_predict_end(self) -> None:
        """Redirect ``on_prediction_end``."""
        if self.model_name == "OVAnomalyModel":
            return  # Ignore when it is an OpenVINO model.
        self.model.model.on_predict_end()

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.Optimizer]] | None:  # type: ignore[override]
        """Configure optimizers for Anomalib models.

        If the anomalib lightning model supports optimizers, return the optimizer.
        Else don't return optimizer even if it is configured in the OTX model.
        """
        # [TODO](ashwinvaidya17): Revisit this method
        if self.model.model.configure_optimizers() and self.optimizer and self.model.trainable_model:
            optimizer = self.optimizer
            if isinstance(optimizer, list):
                if len(optimizer) > 1:
                    msg = "Only one optimizer should be passed"
                    raise ValueError(msg)
                optimizer = optimizer[0]
            params = getattr(self.model.model.model, self.model.trainable_model).parameters()
            return optimizer(params=params)
        # The provided model does not require optimization
        return None

    def configure_callbacks(self) -> Callback | None:
        """Get all necessary callbacks required for training and post-processing on Anomalib models."""
        if self.model_name == "OVAnomalyModel":
            return None
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

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        """Route optimizer step to anomalib's lightning model's optimizer step."""
        return self.model.model.optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def forward(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | AnomalyResult:
        """Wrap forward method of the Anomalib model."""
        if self.model_name == "OVAnomalyModel":
            outputs = self.model(inputs)
        else:
            _inputs: dict = self._customize_inputs(inputs)
            outputs = self.model.model.forward(_inputs)
            outputs = self._customize_outputs(outputs=outputs, inputs=inputs)
        return outputs

    def state_dict(self) -> dict[str, Any]:
        """Set keys of state_dict to allow correct loading of the model."""
        state_dict = super().state_dict()
        # reorder keys
        extra_info_keys = ("image_threshold_class", "pixel_threshold_class", "normalization_class")
        for key in extra_info_keys:
            if key in state_dict:
                state_dict[f"model.model.{key}"] = state_dict[key]
        for key in extra_info_keys:
            if key in state_dict:
                state_dict.pop(key)

        return state_dict

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state_dict and initialize the Anomalib model.

        Also assigns the keys that were removed when saving the state_dict to save disk space.
        """
        # When engine.predict is called, "state_dict" is passed to the model instead of the entire checkpoint
        # Hence we need ``ckpt.get``
        ckpt = ckpt.get("state_dict", ckpt)

        self._setup_anomalib_lightning_model()
        # extract anomaly_lightning_model's state_dict from ckpt and load it
        anomaly_lightning_module_keys = [key for key in ckpt if key.startswith("model.model")]
        anomaly_lightning_module_state_dict = {}
        for key, value in ckpt.items():
            if key in anomaly_lightning_module_keys:
                anomaly_lightning_module_state_dict[key.split("model.model.")[1]] = value

        self.model.model.load_state_dict(anomaly_lightning_module_state_dict)

        # remove extra info keys
        extra_info_keys = ("image_threshold_class", "pixel_threshold_class", "normalization_class", "_is_fine_tuned")
        for key in extra_info_keys:
            if f"model.model.{key}" in ckpt:
                ckpt.pop(f"model.model.{key}")

        return super().load_state_dict(ckpt, *args, **kwargs)

    @abstractmethod
    def _customize_inputs(self, inputs: T_OTXBatchDataEntity) -> dict[str, Any]:  # anomalib model inputs
        """Customize inputs for the model."""
        raise NotImplementedError

    @abstractmethod
    def _customize_outputs(
        self,
        outputs: dict,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity:
        raise NotImplementedError


class OTXAnomalyClassificationLitModel(
    OTXBaseAnomalyLitModel[AnomalyClassificationBatchPrediction, AnomalyClassificationDataBatch],
):
    """OTX Anomaly Classification lightning model."""

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

    def _customize_outputs(
        self,
        outputs: dict,
        inputs: AnomalyClassificationDataBatch,
    ) -> AnomalyClassificationBatchPrediction:
        return AnomalyClassificationBatchPrediction(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["label"],
            # Note: this is the anomalous score. It should be inverted to report Normal score
            scores=outputs["pred_scores"],
            anomaly_maps=outputs["anomaly_maps"],
        )


class OTXAnomalySegmentationLitModel(
    OTXBaseAnomalyLitModel[AnomalySegmentationBatchPrediction, AnomalySegmentationDataBatch],
):
    """OTX Anomaly Segmentation lightning model."""

    def __init__(
        self,
        otx_model: OTXAnomalyModel,
        torch_compile: bool,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
    ):
        super().__init__(otx_model, torch_compile, optimizer, scheduler, task_type=TaskType.SEGMENTATION)

    def _customize_inputs(self, inputs: AnomalySegmentationDataBatch) -> dict[str, Any]:  # anomalib model inputs
        """Customize inputs for the model."""
        return {"image": inputs.images, "label": torch.vstack(inputs.labels).squeeze(), "mask": inputs.masks}

    def _customize_outputs(
        self,
        outputs: dict,
        inputs: AnomalySegmentationDataBatch,
    ) -> AnomalySegmentationBatchPrediction:
        return AnomalySegmentationBatchPrediction(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["label"],
            # Note: this is the anomalous score. It should be inverted to report Normal score
            scores=outputs["pred_scores"],
            anomaly_maps=outputs["anomaly_maps"],
            masks=outputs["mask"],
        )


class OTXAnomalyDetectionLitModel(
    OTXBaseAnomalyLitModel[AnomalyDetectionBatchPrediction, AnomalyDetectionDataBatch],
):
    """OTX Anomaly Detection lightning model."""

    def __init__(
        self,
        otx_model: OTXAnomalyModel,
        torch_compile: bool,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
    ):
        super().__init__(otx_model, torch_compile, optimizer, scheduler, task_type=TaskType.DETECTION)

    def _customize_inputs(self, inputs: AnomalyDetectionDataBatch) -> dict[str, Any]:  # anomalib model inputs
        """Customize inputs for the model."""
        return {
            "image": inputs.images,
            "label": torch.vstack(inputs.labels).squeeze(),
            "mask": inputs.masks,
            "boxes": inputs.boxes,
        }

    def _customize_outputs(self, outputs: dict, inputs: AnomalyDetectionDataBatch) -> AnomalyDetectionBatchPrediction:
        return AnomalyDetectionBatchPrediction(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["label"],
            # Note: this is the anomalous score. It should be inverted to report Normal score
            scores=outputs["pred_scores"],
            anomaly_maps=outputs["anomaly_maps"],
            masks=outputs["mask"],
            boxes=outputs["pred_boxes"],
            box_scores=outputs["box_scores"],
            box_labels=outputs["box_labels"],
        )
