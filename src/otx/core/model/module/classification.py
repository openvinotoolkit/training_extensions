# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification lightning module used in OTX."""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from otx.core.metrices.accuracy import (
    CustomMulticlassAccuracy,
    CustomMultilabelAccuracy,
    CustomHlabelAccuracy
)

from otx.core.metrices.hlabel_accuracy import HLabelAccuracy
from otx.core.data.dataset.classification import HLabelMetaInfo
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
from otx.core.model.entity.classification import OTXHlabelClsModel, OTXMulticlassClsModel, OTXMultilabelClsModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torchmetrics import Metric
    
    from otx.core.metrices import MetricCallable
    from otx.core.data.dataset.base import LabelInfo


class OTXMulticlassClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX multi-class classification task."""

    def __init__(
        self,
        otx_model: OTXMulticlassClsModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable | None = None,
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )

        if metric:
            sig = inspect.signature(metric)
            param_dict = {}
            for name, param in sig.parameters.items():
                param_dict[name] = param.default if name != "num_classes" else self.model.num_classes
            param_dict.pop("kwargs")

            metric = metric(**param_dict)  # type: ignore[call-arg]

        self.metric = metric
    
    def _log_metrics(self, meter: Metric, key: str) -> None:
        results = meter.compute()["accuracy"]
        if results is None:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

        self.log(f"{key}/accuracy", results['accuracy'].item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: MulticlassClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, (MulticlassClsBatchPredEntity, MulticlassClsBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if self.metric:
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

        if self.metric:
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )

    def on_fit_start(self) -> None:
        self.metric = CustomMulticlassAccuracy(average="MICRO", label_info=self.model.label_info).to(self.device)
        
        
class OTXMultilabelClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX multi-label classification task."""

    def __init__(
        self,
        otx_model: OTXMultilabelClsModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable | None = None,
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )
        if metric:
            sig = inspect.signature(metric)
            param_dict = {}
            for name, param in sig.parameters.items():
                param_dict[name] = param.default if name != "num_labels" else self.model.num_classes
            param_dict.pop("kwargs")

            metric = metric(**param_dict)  # type: ignore[call-arg]
        self.metric = metric

    def _log_metrics(self, meter: Metric, key: str) -> None:
        results = meter.compute()["accuracy"]
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

        if self.metric:
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

        if self.metric:
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )
    
    def on_fit_start(self) -> None:
        self.metric = CustomMultilabelAccuracy(average="MICRO", label_info=self.model.label_info).to(self.device)


class OTXHlabelClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX H-label classification task."""

    def __init__(
        self,
        otx_model: OTXHlabelClsModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable | None = None,
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )

        # Temporary, TODO (sungmanc)
        # OTX can't support the auto-configuration for the H-label classification
        # Therefore, default metric can be None and it cause the error
        # This is the workaround
        self.metric = HLabelAccuracy(
            num_multiclass_heads=self.model.num_multiclass_heads,
            num_multilabel_classes=self.model.num_multilabel_classes,
        )

    def _set_hlabel_setup(self) -> None:
        if not isinstance(self.model.label_info, HLabelMetaInfo):
            msg = f"The type of self.label_info should be HLabelMetaInfo, got {type(self.label_info)}."
            raise TypeError(msg)

        self.hlabel_info = self.model.label_info.hlabel_info

        # Set the OTXHlabelClsModel params to make proper hlabel setup.
        self.model.set_hlabel_info(self.hlabel_info)

        # Set the OTXHlabelClsLitModule params.
        self.num_labels = len(self.model.label_info.label_names)
        self.num_multiclass_heads = self.hlabel_info.num_multiclass_heads
        self.num_multilabel_classes = self.hlabel_info.num_multilabel_classes
        self.num_singlelabel_classes = self.num_labels - self.num_multilabel_classes

        # self.val_metric = HLabelAccuracy(
        #     num_multiclass_heads=self.num_multiclass_heads,
        #     num_multilabel_classes=self.num_multilabel_classes,
        #     head_idx_to_logits_range=self.hlabel_info.head_idx_to_logits_range,
        # )
        # self.test_metric = HLabelAccuracy(
        #     num_multiclass_heads=self.num_multiclass_heads,
        #     num_multilabel_classes=self.num_multilabel_classes,
        #     head_idx_to_logits_range=self.hlabel_info.head_idx_to_logits_range,
        # )
        
        self.metric = CustomHlabelAccuracy(average="MICRO", label_info=self.label_info) 

    def _log_metrics(self, meter: Metric, key: str) -> None:
        results = meter.compute()["accuracy"]
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

        if self.metric:
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

        if self.metric:
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )

    @property
    def label_info(self) -> LabelInfo:
        """Meta information of OTXLitModule."""
        return self.model.label_info

    @label_info.setter
    def label_info(self, label_info: LabelInfo) -> None:
        self.model.label_info = label_info
        self._set_hlabel_setup()

    def on_fit_start(self) -> None:
        self.metric = CustomHlabelAccuracy(average="MICRO", label_info=self.model.label_info).to(self.device)