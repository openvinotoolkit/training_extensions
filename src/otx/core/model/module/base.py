# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for base lightning module used in OTX."""
from __future__ import annotations

import inspect
import logging
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from lightning import LightningModule
from torch import Tensor
from torchmetrics import Metric

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchLossEntity,
    OTXBatchPredEntity,
)
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.utils.utils import is_ckpt_for_finetuning, is_ckpt_from_otx_v1

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.data.dataset.base import LabelInfo
    from otx.core.metrics import MetricCallable


class OTXLitModule(LightningModule):
    """Base class for the lightning module used in OTX."""

    def __init__(
        self,
        *,
        otx_model: OTXModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable = lambda: Metric(),
    ):
        super().__init__()

        self.model = otx_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.torch_compile = torch_compile
        self.metric_callable = metric

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["otx_model"])

    def training_step(self, inputs: OTXBatchDataEntity, batch_idx: int) -> Tensor:
        """Step for model training."""

        train_loss = self.model(inputs)
        print(train_loss[list(train_loss.keys())[0]].device, inputs.images.device, self.device)

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

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        self.configure_metric()

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        self.configure_metric()

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        if isinstance(self.metric, Metric):
            self.metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        if isinstance(self.metric, Metric):
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

        self.model.setup_callback(self.trainer)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        optimizers = [
            optimizer(params=self.parameters()) if callable(optimizer) else optimizer
            for optimizer in ensure_list(self.hparams.optimizer)
        ]

        lr_schedulers = []
        for scheduler_config in ensure_list(self.hparams.scheduler):
            scheduler = scheduler_config(optimizers[0]) if callable(scheduler_config) else scheduler_config
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_schedulers.append(lr_scheduler_config)

        return optimizers, lr_schedulers

    def configure_metric(self) -> None:
        """Configure the metric."""
        if isinstance(self.metric_callable, partial):
            num_classes_augmented_params = {
                name: param.default if name != "num_classes" else self.model.num_classes
                for name, param in inspect.signature(self.metric_callable).parameters.items()
                if name != "kwargs"
            }
            self.metric = self.metric_callable(**num_classes_augmented_params)

        if isinstance(self.metric_callable, Metric):
            self.metric = self.metric_callable

        if not isinstance(self.metric, Metric):
            msg = "Metric should be the instance of torchmetrics.Metric."
            raise TypeError(msg)
        self.metric.to(self.device)

    def register_load_state_dict_pre_hook(self, model_classes: list[str], ckpt_classes: list[str]) -> None:
        """Register self.model's load_state_dict_pre_hook.

        Args:
            model_classes (list[str]): Class names from training data.
            ckpt_classes (list[str]): Class names from checkpoint state dictionary.
        """
        self.model.register_load_state_dict_pre_hook(model_classes, ckpt_classes)

    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary of model entity with meta information.

        Returns:
            A dictionary containing datamodule state.

        """
        state_dict = super().state_dict()
        state_dict["label_info"] = self.label_info
        return state_dict

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dictionary from checkpoint state dictionary.

        It successfully loads the checkpoint from OTX v1.x and for finetune and for resume.

        If checkpoint's label_info and OTXLitModule's label_info are different,
        load_state_pre_hook for smart weight loading will be registered.
        """
        if is_ckpt_from_otx_v1(ckpt):
            msg = "The checkpoint comes from OTXv1, checkpoint keys will be updated automatically."
            warnings.warn(msg, stacklevel=2)
            state_dict = self.model.load_from_otx_v1_ckpt(ckpt)
        elif is_ckpt_for_finetuning(ckpt):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        ckpt_label_info = state_dict.pop("label_info", None)

        if ckpt_label_info and self.label_info is None:
            msg = (
                "`state_dict` to load has `label_info`, but the current model has no `label_info`. "
                "It is recommended to set proper `label_info` for the incremental learning case."
            )
            warnings.warn(msg, stacklevel=2)
        if ckpt_label_info and self.label_info and ckpt_label_info != self.label_info:
            logger = logging.getLogger()
            logger.info(
                f"Data classes from checkpoint: {ckpt_label_info.label_names} -> "
                f"Data classes from training data: {self.label_info.label_names}",
            )
            self.register_load_state_dict_pre_hook(
                self.label_info.label_names,
                ckpt_label_info.label_names,
            )
        return super().load_state_dict(state_dict, *args, **kwargs)

    @property
    def label_info(self) -> LabelInfo:
        """Get the member `OTXModel` label information."""
        return self.model.label_info

    @label_info.setter
    def label_info(self, label_info: LabelInfo | list[str]) -> None:
        """Set the member `OTXModel` label information."""
        self.model.label_info = label_info  # type: ignore[assignment]

    def forward(self, *args, **kwargs) -> OTXBatchPredEntity | OTXBatchLossEntity:
        """Model forward pass."""
        if self.model.explain_mode and not isinstance(self.model, OVModel):
            return self.model.forward_explain(*args, **kwargs)
        return self.model.forward(*args, **kwargs)

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model
        Returns:
            Path: path to the exported model.
        """
        return self.model.export(output_dir, base_name, export_format, precision)
