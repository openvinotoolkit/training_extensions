# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for base model entity used in OTX."""

# mypy: disable-error-code="arg-type"

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, NamedTuple, Sequence

import numpy as np
import openvino
import torch
from datumaro import LabelCategories
from jsonargparse import ArgumentParser
from lightning import LightningModule, Trainer
from model_api.models import Model
from model_api.tilers import Tiler
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.sgd import SGD
from torchmetrics import Metric, MetricCollection

from otx import __version__
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import (
    ImageInfo,
    OTXBatchDataEntity,
    OTXBatchLossEntity,
    T_OTXBatchDataEntity,
    T_OTXBatchPredEntity,
)
from otx.core.data.entity.tile import OTXTileBatchDataEntity
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import MetricInput, NullMetricCallable
from otx.core.optimizer.callable import OptimizerCallableSupportHPO
from otx.core.schedulers import (
    LinearWarmupScheduler,
    LinearWarmupSchedulerCallable,
    LRSchedulerListCallable,
    SchedulerCallableSupportHPO,
)
from otx.core.types.export import OTXExportFormatType, TaskLevelExportParameters
from otx.core.types.label import LabelInfo, LabelInfoTypes, NullLabelInfo
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTrainType
from otx.core.utils.build import get_default_num_async_infer_requests
from otx.core.utils.miscellaneous import ensure_callable
from otx.core.utils.utils import is_ckpt_for_finetuning, is_ckpt_from_otx_v1, remove_state_dict_prefix

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
    from model_api.adapters import OpenvinoAdapter
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer, params_t

    from otx.core.data.module import OTXDataModule
    from otx.core.exporter.base import OTXModelExporter
    from otx.core.metrics import MetricCallable

logger = logging.getLogger()


def _default_optimizer_callable(params: params_t) -> Optimizer:
    return SGD(params=params, lr=0.01)


def _default_scheduler_callable(
    optimizer: Optimizer,
    interval: Literal["epoch", "step"] = "epoch",
    **kwargs,
) -> LRScheduler:
    scheduler = ConstantLR(optimizer=optimizer, **kwargs)
    # NOTE: "interval" attribute should be set to configure the scheduler's step interval correctly
    scheduler.interval = interval
    return scheduler


DefaultOptimizerCallable = _default_optimizer_callable
DefaultSchedulerCallable = _default_scheduler_callable


class OTXModel(LightningModule, Generic[T_OTXBatchDataEntity, T_OTXBatchPredEntity]):
    """Base class for the models used in OTX.

    Args:
        num_classes: Number of classes this model can predict.

    Attributes:
        explain_mode: If true, `self.predict_step()` will produce a XAI output as well
        input_size_multiplier (int):
            multiplier value for input size a model requires. If input_size isn't multiple of this value,
            error is raised.
    """

    _OPTIMIZED_MODEL_BASE_NAME: str = "optimized_model"
    input_size_multiplier: int = 1

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] | None = None,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = NullMetricCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        train_type: Literal[OTXTrainType.SUPERVISED, OTXTrainType.SEMI_SUPERVISED] = OTXTrainType.SUPERVISED,
    ) -> None:
        super().__init__()

        self._label_info = self._dispatch_label_info(label_info)
        self.train_type = train_type
        self._check_input_size(input_size)
        self.input_size = input_size
        self.classification_layers: dict[str, dict[str, Any]] = {}
        self.model = self._create_model()
        self.optimizer_callable = ensure_callable(optimizer)
        self.scheduler_callable = ensure_callable(scheduler)
        self.metric_callable = ensure_callable(metric)

        self.torch_compile = torch_compile
        self._explain_mode = False

        # NOTE: To guarantee immutablility of the default value
        self._tile_config = tile_config.clone()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # TODO(vinnamki): Ticket no. 138995: MetricCallable should be saved in the checkpoint
        # so that it can retrieve it from the checkpoint
        self.save_hyperparameters(logger=False, ignore=["optimizer", "scheduler", "metric"])

    def training_step(self, batch: T_OTXBatchDataEntity, batch_idx: int) -> Tensor:
        """Step for model training."""
        train_loss = self.forward(inputs=batch)

        if isinstance(train_loss, Tensor):
            self.log(
                "train/loss",
                train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return train_loss
        if isinstance(train_loss, dict):
            for k, v in train_loss.items():
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

            total_train_loss = sum(train_loss.values())
            self.log(
                "train/loss",
                total_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return total_train_loss

        raise TypeError(train_loss)

    def validation_step(self, batch: T_OTXBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.forward(inputs=batch)

        if isinstance(preds, OTXBatchLossEntity):
            raise TypeError(preds)

        metric_inputs = self._convert_pred_entity_to_compute_metric(preds, batch)

        if isinstance(metric_inputs, dict):
            self.metric.update(**metric_inputs)
            return

        if isinstance(metric_inputs, list) and all(isinstance(inp, dict) for inp in metric_inputs):
            for inp in metric_inputs:
                self.metric.update(**inp)
            return

        raise TypeError(metric_inputs)

    def test_step(self, batch: T_OTXBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.forward(inputs=batch)

        if isinstance(preds, OTXBatchLossEntity):
            raise TypeError(preds)

        metric_inputs = self._convert_pred_entity_to_compute_metric(preds, batch)

        if isinstance(metric_inputs, dict):
            self.metric.update(**metric_inputs)
            return

        if isinstance(metric_inputs, list) and all(isinstance(inp, dict) for inp in metric_inputs):
            for inp in metric_inputs:
                self.metric.update(**inp)
            return

        raise TypeError(metric_inputs)

    def predict_step(
        self,
        batch: T_OTXBatchDataEntity,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> T_OTXBatchPredEntity:
        """Step function called during PyTorch Lightning Trainer's predict."""
        if self.explain_mode:
            return self.forward_explain(inputs=batch)

        outputs = self.forward(inputs=batch)

        if isinstance(outputs, OTXBatchLossEntity):
            raise TypeError(outputs)

        return outputs

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        self.configure_metric()

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        self.configure_metric()

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        self.metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        self.metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        self._log_metrics(self.metric, "val")

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        self._log_metrics(self.metric, "test")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.torch_compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure an optimizer and learning-rate schedulers.

        Configure an optimizer and learning-rate schedulers
        from the given optimizer and scheduler or scheduler list callable in the constructor.
        Generally, there is two lr schedulers. One is for a linear warmup scheduler and
        the other is the main scheduler working after the warmup period.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        optimizer = self.optimizer_callable(self.parameters())
        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    def configure_metric(self) -> None:
        """Configure the metric."""
        if not callable(self.metric_callable):
            raise TypeError(self.metric_callable)

        metric = self.metric_callable(self.label_info)

        if not isinstance(metric, (Metric, MetricCollection)):
            msg = "Metric should be the instance of `torchmetrics.Metric` or `torchmetrics.MetricCollection`."
            raise TypeError(msg, metric)

        self._metric = metric.to(self.device)

    @property
    def metric(self) -> Metric | MetricCollection:
        """Metric module for this OTX model."""
        return self._metric

    @abstractmethod
    def _convert_pred_entity_to_compute_metric(
        self,
        preds: T_OTXBatchPredEntity,
        inputs: T_OTXBatchDataEntity,
    ) -> MetricInput:
        """Convert given inputs to a Python dictionary for the metric computation."""
        raise NotImplementedError

    def _log_metrics(self, meter: Metric, key: Literal["val", "test"], **compute_kwargs) -> None:
        sig = inspect.signature(meter.compute)
        filtered_kwargs = {key: value for key, value in compute_kwargs.items() if key in sig.parameters}
        if removed_kwargs := set(compute_kwargs.keys()).difference(filtered_kwargs.keys()):
            msg = f"These keyword arguments are removed since they are not in the function signature: {removed_kwargs}"
            logger.debug(msg)

        results: dict[str, Tensor] = meter.compute(**filtered_kwargs)

        if not isinstance(results, dict):
            raise TypeError(results)

        if not results:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

        for name, value in results.items():
            log_metric_name = f"{key}/{name}"

            if not isinstance(value, Tensor) or value.numel() != 1:
                msg = f"Log metric name={log_metric_name} is not a scalar tensor. Skip logging it."
                warnings.warn(msg, stacklevel=1)
                continue

            self.log(log_metric_name, value.to(self.device), sync_dist=True, prog_bar=True)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on saving checkpoint."""
        if self.torch_compile:
            # If torch_compile is True, a prefix key named _orig_mod. is added to the state_dict. Remove this.
            compiled_state_dict = checkpoint["state_dict"]
            checkpoint["state_dict"] = remove_state_dict_prefix(compiled_state_dict, "_orig_mod.")
        super().on_save_checkpoint(checkpoint)

        checkpoint["label_info"] = self.label_info
        checkpoint["otx_version"] = __version__
        checkpoint["tile_config"] = self.tile_config

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on loading checkpoint."""
        super().on_load_checkpoint(checkpoint)

        if ckpt_label_info := checkpoint.get("label_info", None):
            self._label_info = ckpt_label_info

        if ckpt_tile_config := checkpoint.get("tile_config", None):
            self.tile_config = ckpt_tile_config

    def load_state_dict_incrementally(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dict incrementally."""
        ckpt_label_info: LabelInfo | None = (
            ckpt.get("label_info", None) if not is_ckpt_from_otx_v1(ckpt) else self.get_ckpt_label_info_v1(ckpt)
        )

        if ckpt_label_info is None:
            msg = "Checkpoint should have `label_info`."
            raise ValueError(msg, ckpt_label_info)

        if ckpt_label_info != self.label_info:
            msg = (
                "Load model state dictionary incrementally: "
                f"Label info from checkpoint: {ckpt_label_info} -> "
                f"Label info from training data: {self.label_info}"
            )
            logger.info(msg)
            self.register_load_state_dict_pre_hook(
                self.label_info.label_names,
                ckpt_label_info.label_names,
            )

        # Model weights
        state_dict: dict[str, Any] = ckpt.get("state_dict", None) if not is_ckpt_from_otx_v1(ckpt) else ckpt

        if state_dict is None:
            msg = "Checkpoint should have `state_dict`."
            raise ValueError(msg, state_dict)

        self.load_state_dict(state_dict, *args, **kwargs)

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dictionary from checkpoint state dictionary.

        It successfully loads the checkpoint from OTX v1.x and for finetune and for resume.

        If checkpoint's label_info and OTXLitModule's label_info are different,
        load_state_pre_hook for smart weight loading will be registered.
        """
        if is_ckpt_from_otx_v1(ckpt):
            msg = "The checkpoint comes from OTXv1, checkpoint keys will be updated automatically."
            warnings.warn(msg, stacklevel=2)
            state_dict = self.load_from_otx_v1_ckpt(ckpt)
        elif is_ckpt_for_finetuning(ckpt):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        return super().load_state_dict(state_dict, *args, **kwargs)

    def load_from_otx_v1_ckpt(self, ckpt: dict[str, Any]) -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        raise NotImplementedError

    @staticmethod
    def get_ckpt_label_info_v1(ckpt: dict) -> LabelInfo:
        """Generate label info from OTX v1 checkpoint."""
        return LabelInfo.from_dm_label_groups(LabelCategories.from_iterable(ckpt["labels"].keys()))

    @property
    def label_info(self) -> LabelInfo:
        """Get this model label information."""
        return self._label_info

    @label_info.setter
    def label_info(self, label_info: LabelInfoTypes) -> None:
        """Set this model label information."""
        self._set_label_info(label_info)

    def _set_label_info(self, label_info: LabelInfoTypes) -> None:
        """Actual implementation for set this model label information.

        Derived classes should override this function.
        """
        msg = (
            "Assign new label_info to the model. "
            "It is usually not recommended. "
            "Please create a new model instance by giving label_info to its initializer "
            "such as `OTXModel(label_info=label_info, ...)`."
        )
        logger.warning(msg, stacklevel=0)

        new_label_info = self._dispatch_label_info(label_info)

        old_num_classes = self._label_info.num_classes
        new_num_classes = new_label_info.num_classes

        if old_num_classes != new_num_classes:
            msg = (
                f"Given LabelInfo has the different number of classes "
                f"({old_num_classes}!={new_num_classes}). "
                "The model prediction layer is reset to the new number of classes "
                f"(={new_num_classes})."
            )
            logger.warning(msg, stacklevel=0)
            self._reset_prediction_layer(num_classes=new_label_info.num_classes)

        self._label_info = new_label_info

    @property
    def num_classes(self) -> int:
        """Returns model's number of classes. Can be redefined at the model's level."""
        return self.label_info.num_classes

    @property
    def explain_mode(self) -> bool:
        """Get model explain mode."""
        return self._explain_mode

    @explain_mode.setter
    def explain_mode(self, explain_mode: bool) -> None:
        """Set model explain mode."""
        self._explain_mode = explain_mode

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""

    def _customize_inputs(self, inputs: T_OTXBatchDataEntity) -> dict[str, Any]:
        """Customize OTX input batch data entity if needed for your model."""
        raise NotImplementedError

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for model."""
        raise NotImplementedError

    def forward(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""
        # If customize_inputs is overridden
        if isinstance(inputs, OTXTileBatchDataEntity):
            return self.forward_tiles(inputs)

        outputs = (
            self.model(**self._customize_inputs(inputs))
            if self._customize_inputs != OTXModel._customize_inputs
            else self.model(inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != OTXModel._customize_outputs
            else outputs
        )

    def forward_explain(self, inputs: T_OTXBatchDataEntity) -> T_OTXBatchPredEntity:
        """Model forward explain function."""
        msg = "Derived model class should implement this class to support the explain pipeline."
        raise NotImplementedError(msg)

    def forward_for_tracing(self, *args, **kwargs) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        msg = (
            "Derived model class should implement this class to support the export pipeline. "
            "If it wants to use `otx.core.exporter.native.OTXNativeModelExporter`."
        )
        raise NotImplementedError(msg)

    def get_explain_fn(self) -> Callable:
        """Returns explain function."""
        raise NotImplementedError

    def forward_tiles(
        self,
        inputs: OTXTileBatchDataEntity[T_OTXBatchDataEntity],
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Model forward function for tile task."""
        raise NotImplementedError

    def register_load_state_dict_pre_hook(self, model_classes: list[str], ckpt_classes: list[str]) -> None:
        """Register load_state_dict_pre_hook.

        Args:
            model_classes (list[str]): Class names from training data.
            ckpt_classes (list[str]): Class names from checkpoint state dictionary.
        """
        self.model_classes = model_classes
        self.ckpt_classes = ckpt_classes
        self._register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)

    def load_state_dict_pre_hook(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs) -> None:
        """Modify input state_dict according to class name matching before weight loading."""
        model2ckpt = self.map_class_names(self.model_classes, self.ckpt_classes)

        for param_name, info in self.classification_layers.items():
            model_param = self.state_dict()[param_name].clone()
            ckpt_param = state_dict[prefix + param_name]
            stride = info.get("stride", 1)
            num_extra_classes = info.get("num_extra_classes", 0)
            for model_dst, ckpt_dst in enumerate(model2ckpt):
                if ckpt_dst >= 0:
                    model_param[(model_dst) * stride : (model_dst + 1) * stride].copy_(
                        ckpt_param[(ckpt_dst) * stride : (ckpt_dst + 1) * stride],
                    )
            if num_extra_classes > 0:
                num_ckpt_class = len(self.ckpt_classes)
                num_model_class = len(self.model_classes)
                model_param[(num_model_class) * stride : (num_model_class + 1) * stride].copy_(
                    ckpt_param[(num_ckpt_class) * stride : (num_ckpt_class + 1) * stride],
                )

            # Replace checkpoint weight by mixed weights
            state_dict[prefix + param_name] = model_param

    @staticmethod
    def map_class_names(src_classes: list[str], dst_classes: list[str]) -> list[int]:
        """Computes src to dst index mapping.

        src2dst[src_idx] = dst_idx
        #  according to class name matching, -1 for non-matched ones
        assert(len(src2dst) == len(src_classes))
        ex)
          src_classes = ['person', 'car', 'tree']
          dst_classes = ['tree', 'person', 'sky', 'ball']
          -> Returns src2dst = [1, -1, 0]
        """
        src2dst = []
        for src_class in src_classes:
            if src_class in dst_classes:
                src2dst.append(dst_classes.index(src_class))
            else:
                src2dst.append(-1)
        return src2dst

    def optimize(self, output_dir: Path, data_module: OTXDataModule, ptq_config: dict[str, Any] | None = None) -> Path:
        """Runs quantization of the model with NNCF.PTQ on the passed data. Works only for OpenVINO models.

        PTQ performs int-8 quantization on the input model, so the resulting model
        comes in mixed precision (some operations, however, remain in FP32).

        Args:
            output_dir (Path): working directory to save the optimized model.
            data_module (OTXDataModule): dataset for calibration of quantized layers.
            ptq_config (dict[str, Any] | None): config for NNCF.PTQ.

        Returns:
            Path: path to the resulting optimized OpenVINO model.
        """
        msg = "Optimization is not implemented for torch models"
        raise NotImplementedError(msg)

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        to_exportable_code: bool = False,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model
            to_exportable_code (bool): flag to export model in exportable code with demo package

        Returns:
            Path: path to the exported model.
        """
        mode = self.training
        self.eval()

        orig_forward = self.forward
        orig_trainer = self._trainer  # type: ignore[has-type]
        try:
            if self._trainer is None:  # type: ignore[has-type]
                self._trainer = Trainer()
            self.forward = self.forward_for_tracing  # type: ignore[method-assign, assignment]
            return self._exporter.export(
                self,
                output_dir,
                base_name,
                export_format,
                precision,
                to_exportable_code,
            )
        finally:
            self.train(mode)
            self.forward = orig_forward  # type: ignore[method-assign]
            self._trainer = orig_trainer

    @property
    def _exporter(self) -> OTXModelExporter:
        """Defines exporter of the model. Should be overridden in subclasses."""
        msg = (
            "To export this OTXModel, you should implement an appropriate exporter for it. "
            "You can try to reuse ones provided in `otx.core.exporter.*`."
        )
        raise NotImplementedError(msg)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines export parameters sharable at a task level.

        To export OTXModel which is compatible with ModelAPI,
        you should define an appropriate export parameters for each task.
        This property is usually defined at the task level classes defined in `otx.core.model.*`.
        Please refer to `TaskLevelExportParameters` for more details.

        Returns:
            Collection of exporter parameters that can be defined at a task level.

        Examples:
            This example shows how this property is used at the new model development

            ```python

            class MyDetectionModel(OTXDetectionModel):
                ...

                @property
                def _exporter(self) -> OTXModelExporter:
                    # `self._export_parameters` defined at `OTXDetectionModel`
                    # You can redefine it `MyDetectionModel` if you need
                    return OTXModelExporter(
                        task_level_export_parameters=self._export_parameters,
                        ...
                    )
            ```
        """
        return TaskLevelExportParameters(
            model_type="null",
            task_type="null",
            label_info=self.label_info,
            optimization_config=self._optimization_config,
        )

    def _reset_prediction_layer(self, num_classes: int) -> None:
        """Reset its prediction layer with a given number of classes.

        Args:
            num_classes: Number of classes
        """
        raise NotImplementedError

    @property
    def _optimization_config(self) -> dict[str, str]:
        return {}

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Tensor) -> None:
        """It is required to prioritize the warmup lr scheduler than other lr scheduler during a warmup period.

        It will ignore other lr scheduler's stepping if the warmup scheduler is currently activated.
        """
        warmup_schedulers = [
            config.scheduler
            for config in self.trainer.lr_scheduler_configs
            if isinstance(config.scheduler, LinearWarmupScheduler)
        ]

        if not warmup_schedulers:
            # There is no warmup scheduler
            return super().lr_scheduler_step(scheduler=scheduler, metric=metric)

        if len(warmup_schedulers) != 1:
            msg = "No more than two warmup schedulers coexist."
            raise RuntimeError(msg)

        warmup_scheduler = next(iter(warmup_schedulers))

        if scheduler != warmup_scheduler and warmup_scheduler.activated:
            msg = (
                "Warmup lr scheduler is currently activated. "
                "Ignore other schedulers until the warmup lr scheduler is finished"
            )
            logger.debug(msg)
            return None

        return super().lr_scheduler_step(scheduler=scheduler, metric=metric)

    def patch_optimizer_and_scheduler_for_hpo(self) -> None:
        """Patch optimizer and scheduler for hyperparameter optimization and adaptive batch size.

        This is inplace function changing inner states (`optimizer_callable` and `scheduler_callable`).
        Both will be changed to be picklable. In addition, `optimizer_callable` is changed
        to make its hyperparameters gettable.
        """
        if not isinstance(self.optimizer_callable, OptimizerCallableSupportHPO):
            self.optimizer_callable = OptimizerCallableSupportHPO.from_callable(self.optimizer_callable)

        if not isinstance(self.scheduler_callable, SchedulerCallableSupportHPO) and not isinstance(
            self.scheduler_callable,
            LinearWarmupSchedulerCallable,  # LinearWarmupSchedulerCallable natively supports HPO
        ):
            self.scheduler_callable = SchedulerCallableSupportHPO.from_callable(self.scheduler_callable)

    @property
    def tile_config(self) -> TileConfig:
        """Get tiling configurations."""
        return self._tile_config

    @tile_config.setter
    def tile_config(self, tile_config: TileConfig) -> None:
        """Set tiling configurations."""
        msg = (
            "Assign new tile_config to the model. "
            "It is usually not recommended. "
            "Please create a new model instance by giving tile_config to its initializer "
            "such as `OTXModel(..., tile_config=tile_config)`."
        )
        logger.warning(msg, stacklevel=0)

        self._tile_config = tile_config

    def get_dummy_input(self, batch_size: int = 1) -> OTXBatchDataEntity[Any]:
        """Generates a dummy input, suitable for launching forward() on it.

        Args:
            batch_size (int, optional): number of elements in a dummy input sequence. Defaults to 1.

        Returns:
            OTXBatchDataEntity[Any]: An entity containing randomly generated inference data.
        """
        raise NotImplementedError

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> LabelInfo:
        if isinstance(label_info, int):
            return LabelInfo.from_num_classes(num_classes=label_info)
        if isinstance(label_info, Sequence) and all(isinstance(name, str) for name in label_info):
            return LabelInfo(label_names=label_info, label_groups=[label_info])
        if isinstance(label_info, LabelInfo):
            return label_info

        raise TypeError(label_info)

    def _check_input_size(self, input_size: tuple[int, int] | None = None) -> None:
        if input_size is not None and (
            input_size[0] % self.input_size_multiplier != 0 or input_size[1] % self.input_size_multiplier != 0
        ):
            msg = f"Input size should be a multiple of {self.input_size_multiplier}, but got {input_size} instead."
            raise ValueError(msg)


class OVModel(OTXModel, Generic[T_OTXBatchDataEntity, T_OTXBatchPredEntity]):
    """Base class for the OpenVINO model.

    This is a base class representing interface for interacting with OpenVINO
    Intermediate Representation (IR) models. OVModel can create and validate
    OpenVINO IR model directly from provided path locally or from
    OpenVINO OMZ repository. (Only PyTorch models are supported).
    OVModel supports synchronous as well as asynchronous inference type.

    Args:
        num_classes: Number of classes this model can predict.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        async_inference: bool = True,
        force_cpu: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = NullMetricCallable,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.force_cpu = force_cpu
        self.async_inference = async_inference
        self.num_requests = max_num_requests if max_num_requests is not None else get_default_num_async_infer_requests()
        self.use_throughput_mode = use_throughput_mode
        self.model_api_configuration = model_api_configuration if model_api_configuration is not None else {}
        # NOTE: num_classes and label_info comes from the IR metadata
        super().__init__(label_info=NullLabelInfo(), metric=metric)
        self._label_info = self._create_label_info_from_ov_ir()

        tile_enabled = False
        with contextlib.suppress(RuntimeError):
            if isinstance(self.model, Model):
                tile_enabled = "tile_size" in self.model.inference_adapter.get_rt_info(["model_info"]).astype(dict)

        if tile_enabled:
            self._setup_tiler()

    def _setup_tiler(self) -> None:
        """Setup tiler for tile task."""
        raise NotImplementedError

    def _get_hparams_from_adapter(self, model_adapter: OpenvinoAdapter) -> None:
        """Reads model configuration from ModelAPI OpenVINO adapter.

        Args:
            model_adapter (OpenvinoAdapter): target adapter to read the config
        """

    def _create_model(self) -> Model:
        """Create a OV model with help of Model API."""
        from model_api.adapters import OpenvinoAdapter, create_core

        if self.device.type != "cpu":
            msg = (
                f"Device {self.device.type} is set for Lightning module, but the actual inference "
                "device is selected by OpenVINO."
            )
            logger.warning(msg)

        ov_device = "CPU"
        ie = create_core()
        if not self.force_cpu:
            devices = ie.available_devices
            for device in devices:
                device_name = ie.get_property(device_name=device, property="FULL_DEVICE_NAME")
                if "dGPU" in device_name and "Intel" in device_name:
                    ov_device = device
                    break

        plugin_config = {}
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_adapter = OpenvinoAdapter(
            ie,
            self.model_name,
            device=ov_device,
            max_num_requests=self.num_requests,
            plugin_config=plugin_config,
            model_parameters=self.model_adapter_parameters,
        )

        self._get_hparams_from_adapter(model_adapter)

        return Model.create_model(model_adapter, model_type=self.model_type, configuration=self.model_api_configuration)

    def _customize_inputs(self, entity: T_OTXBatchDataEntity) -> dict[str, Any]:
        # restore original numpy image
        images = [np.transpose(im.cpu().numpy(), (1, 2, 0)) for im in entity.images]
        return {"inputs": images}

    def _forward(self, inputs: T_OTXBatchDataEntity) -> T_OTXBatchPredEntity:
        """Model forward function."""

        def _callback(result: NamedTuple, idx: int) -> None:
            output_dict[idx] = result

        numpy_inputs = self._customize_inputs(inputs)["inputs"]
        if self.async_inference:
            output_dict: dict[int, NamedTuple] = {}
            self.model.set_callback(_callback)
            for idx, im in enumerate(numpy_inputs):
                if not self.model.is_ready():
                    self.model.await_any()
                self.model.infer_async(im, user_data=idx)
            self.model.await_all()
            outputs = [out[1] for out in sorted(output_dict.items())]
        else:
            outputs = [self.model(im) for im in numpy_inputs]

        customized_outputs = self._customize_outputs(outputs, inputs)

        if isinstance(customized_outputs, OTXBatchLossEntity):
            raise TypeError(customized_outputs)

        return customized_outputs

    def forward(self, inputs: T_OTXBatchDataEntity) -> T_OTXBatchPredEntity:
        """Model forward function."""
        return self._forward(inputs=inputs)  # type: ignore[return-value]

    def forward_explain(self, inputs: T_OTXBatchDataEntity) -> T_OTXBatchPredEntity:
        """Model forward explain function."""
        return self._forward(inputs=inputs)  # type: ignore[return-value]

    def optimize(
        self,
        output_dir: Path,
        data_module: OTXDataModule,
        ptq_config: dict[str, Any] | None = None,
    ) -> Path:
        """Runs NNCF quantization."""
        import nncf

        output_model_path = output_dir / (self._OPTIMIZED_MODEL_BASE_NAME + ".xml")

        def check_if_quantized(model: openvino.Model) -> bool:
            """Checks if OpenVINO model is already quantized."""
            nodes = model.get_ops()
            return any(op.get_type_name() == "FakeQuantize" for op in nodes)

        ov_model = openvino.Core().read_model(self.model_name)

        if check_if_quantized(ov_model):
            msg = "Model is already optimized by PTQ"
            raise RuntimeError(msg)

        train_dataset = data_module.train_dataloader()

        ptq_config_from_ir = self._read_ptq_config_from_ir(ov_model)
        if ptq_config is not None:
            ptq_config_from_ir.update(ptq_config)
            ptq_config = ptq_config_from_ir
        else:
            ptq_config = ptq_config_from_ir

        quantization_dataset = nncf.Dataset(train_dataset, self.transform_fn)  # type: ignore[attr-defined]

        compressed_model = nncf.quantize(  # type: ignore[attr-defined]
            ov_model,
            quantization_dataset,
            **ptq_config,
        )

        openvino.save_model(compressed_model, output_model_path)

        return output_model_path

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        to_exportable_code: bool = True,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model
            to_exportable_code (bool): whether to generate exportable code with demo package.
                OpenVINO model supports only exportable code option.

        Returns:
            Path: path to the exported model.
        """
        if not to_exportable_code:
            msg = "OpenVINO model can be exported only as exportable code with demo package."
            raise RuntimeError(msg)

        return self._exporter.export(
            self.model,
            output_dir,
            base_name,
            export_format,
            precision,
            to_exportable_code,
        )

    def transform_fn(self, data_batch: T_OTXBatchDataEntity) -> np.array:
        """Data transform function for PTQ."""
        np_data = self._customize_inputs(data_batch)
        image = np_data["inputs"][0]
        # NOTE: Tiler wraps the model, so we need to unwrap it to get the model
        model = self.model.model if isinstance(self.model, Tiler) else self.model
        resized_image = model.resize(image, (model.w, model.h))
        resized_image = model.input_transform(resized_image)
        return model._change_layout(resized_image)  # noqa: SLF001

    def _read_ptq_config_from_ir(self, ov_model: Model) -> dict[str, Any]:
        """Generates the PTQ (Post-Training Quantization) configuration from the meta data of the given OpenVINO model.

        Args:
            ov_model (Model): The OpenVINO model in which the PTQ configuration is embedded.

        Returns:
            dict: The PTQ configuration as a dictionary.
        """
        from nncf import IgnoredScope  # type: ignore[attr-defined]
        from nncf.common.quantization.structs import QuantizationPreset  # type: ignore[attr-defined]
        from nncf.parameters import ModelType
        from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

        if "optimization_config" not in ov_model.rt_info["model_info"]:
            return {}

        initial_ptq_config = json.loads(ov_model.rt_info["model_info"]["optimization_config"].value)
        if not initial_ptq_config:
            return {}
        argparser = ArgumentParser()
        if "advanced_parameters" in initial_ptq_config:
            argparser.add_class_arguments(AdvancedQuantizationParameters, "advanced_parameters")
        if "preset" in initial_ptq_config:
            initial_ptq_config["preset"] = QuantizationPreset(initial_ptq_config["preset"])
            argparser.add_argument("--preset", type=QuantizationPreset)
        if "model_type" in initial_ptq_config:
            initial_ptq_config["model_type"] = ModelType(initial_ptq_config["model_type"])
            argparser.add_argument("--model_type", type=ModelType)
        if "ignored_scope" in initial_ptq_config:
            argparser.add_class_arguments(IgnoredScope, "ignored_scope", as_positional=True)

        initial_ptq_config = argparser.parse_object(initial_ptq_config)

        return argparser.instantiate_classes(initial_ptq_config).as_dict()

    @property
    def _exporter(self) -> OTXNativeModelExporter:
        """Exporter of the OVModel for exportable code."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, self.model.h, self.model.w),
        )

    @property
    def model_adapter_parameters(self) -> dict:
        """Model parameters for export."""
        return {}

    def _set_label_info(self, label_info: LabelInfoTypes) -> None:
        """Set this model label information."""
        new_label_info = self._dispatch_label_info(label_info)
        self._label_info = new_label_info

    def _create_label_info_from_ov_ir(self) -> LabelInfo:
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            serialized = ov_model.get_rt_info(["model_info", "label_info"]).value
            return LabelInfo.from_json(serialized)

        mapi_model: Model = self.model

        if label_names := getattr(mapi_model, "labels", None):
            msg = (
                'Cannot find "label_info" from OpenVINO IR. '
                "However, we found labels attributes from ModelAPI. "
                "Construct LabelInfo from it."
            )

            logger.warning(msg)
            return LabelInfo(label_names=label_names, label_groups=[label_names])

        msg = "Cannot construct LabelInfo from OpenVINO IR. Please check this model is trained by OTX."
        raise ValueError(msg)

    def get_dummy_input(self, batch_size: int = 1) -> OTXBatchDataEntity:
        """Returns a dummy input for base OV model."""
        # Resize is embedded to the OV model, which means we don't need to know the actual size
        images = [torch.rand(3, 224, 224) for _ in range(batch_size)]
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return OTXBatchDataEntity(batch_size=batch_size, images=images, imgs_info=infos)
