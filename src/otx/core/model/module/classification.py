# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification lightning module used in OTX."""
from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy

from otx.core.data.dataset.classification import HLabelInfo
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    HlabelClsBatchPredEntityWithXAI,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MulticlassClsBatchPredEntityWithXAI,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
    MultilabelClsBatchPredEntityWithXAI,
)
from otx.core.metrics.accuracy import AccuracywithLabelGroup, MixedHLabelAccuracy
from otx.core.model.entity.classification import OTXHlabelClsModel, OTXMulticlassClsModel, OTXMultilabelClsModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.data.dataset.base import LabelInfo
    from otx.core.metrics import MetricCallable


class OTXMulticlassClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX multi-class classification task."""

    def __init__(
        self,
        otx_model: OTXMulticlassClsModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable = lambda num_classes: Accuracy(task="multiclass", num_classes=num_classes),
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()
        if isinstance(self.metric, AccuracywithLabelGroup):
            self.metric.label_info = self.model.label_info

    def _log_metrics(self, meter: Metric, key: str) -> None:
        results = meter.compute()
        if results is None:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

        # Custom Accuracy returns the dictionary, and accuracy value is in the `accuracy` key.
        if isinstance(results, dict):
            results = torch.tensor(results["accuracy"])

        self.log(f"{key}/accuracy", results.item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: MulticlassClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, (MulticlassClsBatchPredEntity, MulticlassClsBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if isinstance(self.metric, Metric):
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MulticlassClsBatchPredEntity | MulticlassClsBatchPredEntityWithXAI,
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

        if not isinstance(preds, (MulticlassClsBatchPredEntity, MulticlassClsBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if isinstance(self.metric, Metric):
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )


class OTXMultilabelClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX multi-label classification task."""

    def __init__(
        self,
        otx_model: OTXMultilabelClsModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable = lambda num_labels: Accuracy(task="multilabel", num_labels=num_labels),
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )

    def configure_metric(self) -> None:
        """Configure the metric."""
        if isinstance(self.metric_callable, partial):
            num_classes_augmented_params = {
                name: param.default if name != "num_labels" else self.model.num_classes
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
        if isinstance(self.metric, AccuracywithLabelGroup):
            self.metric.label_info = self.model.label_info

    def _log_metrics(self, meter: Metric, key: str) -> None:
        results = meter.compute()

        # Custom Accuracy returns the dictionary, and accuracy value is in the `accuracy` key.
        if isinstance(results, dict):
            results = torch.tensor(results["accuracy"])

        self.log(f"{key}/accuracy", results.item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: MultilabelClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, (MultilabelClsBatchPredEntity, MultilabelClsBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if isinstance(self.metric, Metric):
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MultilabelClsBatchPredEntity | MultilabelClsBatchPredEntityWithXAI,
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

        if not isinstance(preds, (MultilabelClsBatchPredEntity, MultilabelClsBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if isinstance(self.metric, Metric):
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )


class OTXHlabelClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX H-label classification task."""

    def __init__(
        self,
        otx_model: OTXHlabelClsModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable = partial(  # noqa: B008
            MixedHLabelAccuracy,
            num_multiclass_heads=2,
            num_multilabel_classes=2,
            head_logits_info={"default": (0, 2)},
        ),  # lambda: MixedHLabelAccuracy() doesn't return the partial class. So, use the partial() directly.
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )

        self.label_info: HLabelInfo
        self.num_labels: int
        self.num_multiclass_heads: int
        self.num_multilabel_classes: int
        self.num_singlelabel_classes: int

    def configure_metric(self) -> None:
        """Configure the metric."""
        if isinstance(self.metric_callable, partial):
            sig = inspect.signature(self.metric_callable)
            param_dict = {}
            for name, param in sig.parameters.items():
                if name in ["num_multiclass_heads", "num_multilabel_classes"]:
                    param_dict[name] = getattr(self.model, name)
                elif name == "head_logits_info" and isinstance(self.label_info, HLabelInfo):
                    param_dict[name] = self.label_info.head_idx_to_logits_range
                else:
                    param_dict[name] = param.default
            param_dict.pop("kwargs", {})
            self.metric = self.metric_callable(**param_dict)
        elif isinstance(self.metric_callable, Metric):
            self.metric = self.metric_callable

        if not isinstance(self.metric, Metric):
            msg = "Metric should be the instance of torchmetrics.Metric."
            raise TypeError(msg)

        self.metric.to(self.device)
        if isinstance(self.metric, AccuracywithLabelGroup):
            self.metric.label_info = self.model.label_info

    def _set_hlabel_setup(self) -> None:
        if not isinstance(self.label_info, HLabelInfo):
            msg = f"The type of self.label_info should be HLabelInfo, got {type(self.label_info)}."
            raise TypeError(msg)

        # Set the OTXHlabelClsModel params to make proper hlabel setup.
        self.model.set_hlabel_info(self.label_info)

        # Set the OTXHlabelClsLitModule params.
        self.num_labels = len(self.label_info.label_names)
        self.num_multiclass_heads = self.label_info.num_multiclass_heads
        self.num_multilabel_classes = self.label_info.num_multilabel_classes
        self.num_singlelabel_classes = self.num_labels - self.num_multilabel_classes

    def _log_metrics(self, meter: Metric, key: str) -> None:
        results = meter.compute()

        # Custom Accuracy returns the dictionary, and accuracy value is in the `accuracy` key.
        if isinstance(results, dict):
            results = torch.tensor(results["accuracy"])

        self.log(f"{key}/accuracy", results.item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: HlabelClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, (HlabelClsBatchPredEntity, HlabelClsBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if isinstance(self.metric, Metric):
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: HlabelClsBatchPredEntity | HlabelClsBatchPredEntityWithXAI,
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

        if not isinstance(preds, (HlabelClsBatchPredEntity, HlabelClsBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if isinstance(self.metric, Metric):
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )

    @property
    def label_info(self) -> LabelInfo:
        """Meta information of OTXLitModule."""
        if self._meta_info is None:
            err_msg = "label_info is referenced before assignment"
            raise TypeError(err_msg)
        return self._meta_info

    @label_info.setter
    def label_info(self, label_info: LabelInfo) -> None:
        self._meta_info = label_info
        self._set_hlabel_setup()