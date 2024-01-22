# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification lightning module used in OTX."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification.accuracy import Accuracy

from otx.algo.classification.metrics import HLabelAccuracy
from otx.core.data.dataset.classification import HLabelMetaInfo
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.model.entity.classification import OTXHlabelClsModel, OTXMulticlassClsModel, OTXMultilabelClsModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.data.dataset.base import LabelInfo


class OTXMulticlassClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX multi-class classification task."""

    def __init__(
        self,
        otx_model: OTXMulticlassClsModel,
        torch_compile: bool,
        optimizer: OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        num_classes = otx_model.config.get("head", {}).get("num_classes", None)
        self.val_metric = Accuracy(task="multiclass", num_classes=num_classes)
        #self.test_metric = Accuracy(task="multiclass", num_classes=num_classes) 

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        self.val_metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        self.test_metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        self._log_metrics(self.val_metric, "val")

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        self._log_metrics(self.test_metric, "test")

    def _log_metrics(self, meter: Accuracy, key: str) -> None:
        results = meter.compute()
        if results is None:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

        self.log(f"{key}/accuracy", results.item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: MulticlassClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, MulticlassClsBatchPredEntity):
            raise TypeError(preds)

        self.val_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MulticlassClsBatchPredEntity,
        inputs: MulticlassClsBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }

    def test_step(self, inputs: MulticlassClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, MulticlassClsBatchPredEntity):
            raise TypeError(preds)

        self.test_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )
    
    def _load_from_prev_otx_ckpt(self, ckpt: dict[str, Any]) -> dict[str, Any]:
        """Get the state_dict, supporting the backward compatibility."""
        return self.model.convert_previous_otx_ckpt(ckpt, add_prefix="model.model.")

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"


class OTXMultilabelClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX multi-label classification task."""

    def __init__(
        self,
        otx_model: OTXMultilabelClsModel,
        torch_compile: bool,
        optimizer: OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.num_labels = otx_model.config.get("head", {}).get("num_classes", None)

        self.val_metric = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5, average="micro")
        self.test_metric = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5, average="micro")

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        self.val_metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        self.test_metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        self._log_metrics(self.val_metric, "val")

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        self._log_metrics(self.test_metric, "test")

    def _log_metrics(self, meter: Accuracy, key: str) -> None:
        results = meter.compute()
        self.log(f"{key}/accuracy", results.item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: MultilabelClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, MultilabelClsBatchPredEntity):
            raise TypeError(preds)

        self.val_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MultilabelClsBatchPredEntity,
        inputs: MultilabelClsBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        return {
            "preds": torch.stack(preds.scores),
            "target": torch.stack(inputs.labels),
        }

    def test_step(self, inputs: MultilabelClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, MultilabelClsBatchPredEntity):
            raise TypeError(preds)

        self.test_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )
    
    def _load_from_prev_otx_ckpt(self, ckpt: dict[str, Any]) -> dict[str, Any]:
        """Get the state_dict, supporting the backward compatibility."""
        return self.model.convert_previous_otx_ckpt(ckpt, add_prefix="model.model.")

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"
    


class OTXHlabelClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX H-label classification task."""

    def __init__(
        self,
        otx_model: OTXHlabelClsModel,
        torch_compile: bool,
        optimizer: OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def _set_hlabel_setup(self) -> None:
        if not isinstance(self.meta_info, HLabelMetaInfo):
            msg = f"The type of self.meta_info should be HLabelMetaInfo, got {type(self.meta_info)}."
            raise TypeError(msg)

        self.hlabel_info = self.meta_info.hlabel_info

        # Set the OTXHlabelClsModel params to make proper hlabel setup.
        self.model.model.head.set_hlabel_info(self.hlabel_info)

        # Set the OTXHlabelClsLitModule params.
        self.num_labels = len(self.meta_info.label_names)
        self.num_multiclass_heads = self.hlabel_info.num_multiclass_heads
        self.num_multilabel_classes = self.hlabel_info.num_multilabel_classes
        self.num_singlelabel_classes = self.num_labels - self.num_multilabel_classes

        self.val_metric = HLabelAccuracy(
            num_multiclass_heads=self.num_multiclass_heads,
            num_multilabel_classes=self.num_multilabel_classes,
            head_idx_to_logits_range=self.hlabel_info.head_idx_to_logits_range,
        )
        self.test_metric = HLabelAccuracy(
            num_multiclass_heads=self.num_multiclass_heads,
            num_multilabel_classes=self.num_multilabel_classes,
            head_idx_to_logits_range=self.hlabel_info.head_idx_to_logits_range,
        )

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        self.val_metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        self.test_metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        self._log_metrics(self.val_metric, "val")

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        self._log_metrics(self.test_metric, "test")

    def _log_metrics(self, meter: Accuracy, key: str) -> None:
        results = meter.compute()
        self.log(f"{key}/accuracy", results.item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: HlabelClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, HlabelClsBatchPredEntity):
            raise TypeError(preds)

        self.val_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: HlabelClsBatchPredEntity,
        inputs: HlabelClsBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        if self.num_multilabel_classes > 0:
            preds_multiclass = torch.stack(preds.labels)[:, : self.num_multiclass_heads]
            preds_multilabel = torch.stack(preds.scores)[:, self.num_multiclass_heads :]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = torch.stack(preds.labels)
        return {
            "preds": pred_result,
            "target": torch.stack(inputs.labels),
        }

    def test_step(self, inputs: HlabelClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, HlabelClsBatchPredEntity):
            raise TypeError(preds)

        self.test_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )
    
    def _load_from_prev_otx_ckpt(self, ckpt: dict[str, Any]) -> dict[str, Any]:
        """Get the state_dict, supporting the backward compatibility."""
        return self.model.convert_previous_otx_ckpt(ckpt, add_prefix="model.model.")

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"

    @property
    def meta_info(self) -> LabelInfo:
        """Meta information of OTXLitModule."""
        if self._meta_info is None:
            err_msg = "meta_info is referenced before assignment"
            raise ValueError(err_msg)
        return self._meta_info

    @meta_info.setter
    def meta_info(self, meta_info: LabelInfo) -> None:
        self._meta_info = meta_info
        self._set_hlabel_setup()
