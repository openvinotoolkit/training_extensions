# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting lightning module used in OTX."""
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, Dice
from torchmetrics.collections import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors

from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.model.entity.visual_prompting import OTXVisualPromptingModel
from otx.core.model.module.base import OTXLitModule
from otx.core.utils.mask_util import polygon_to_bitmap

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torchmetrics import Metric


class OTXVisualPromptingLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX visual prompting task."""

    def __init__(
        self,
        otx_model: OTXVisualPromptingModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: Metric = MeanMetric,  # TODO (sungmanc): dictionary metric will be supported # noqa: TD003
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )

        self.train_metric = MetricCollection(
            {
                "loss": MeanMetric(),
                "loss_dice": MeanMetric(),
                "loss_focal": MeanMetric(),
                "loss_iou": MeanMetric(),
            },
        )

    def configure_metric(self, cond: str = "") -> None:
        """Configure metrics."""
        self.val_metric = MetricCollection(
            {
                "IoU": BinaryJaccardIndex(),
                "F1": BinaryF1Score(),
                "Dice": Dice(),
                "mAP": MeanAveragePrecision(iou_type="segm"),
            },
        )
        self.val_metric.to(self.device)

        self.test_metric = MetricCollection(
            {
                "IoU": BinaryJaccardIndex(),
                "F1": BinaryF1Score(),
                "Dice": Dice(),
                "mAP": MeanAveragePrecision(iou_type="segm"),
            },
        )
        self.test_metric.to(self.device)

    def on_train_epoch_start(self) -> None:
        """Callback triggered when the train epoch starts."""
        self.train_metric.reset()

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        self.val_metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        self.test_metric.reset()

    def on_train_epoch_end(self) -> None:
        """Callback triggered when the train epoch ends."""
        self._log_metrics(self.train_metric, "train")
        self.train_metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        self._log_metrics(self.val_metric, "val")
        self.val_metric.reset()

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        self._log_metrics(self.test_metric, "test")
        self.test_metric.reset()

    def _log_metrics(self, meter: MetricCollection, subset_name: str) -> None:
        results = meter.compute()
        for metric, value in results.items():
            if not isinstance(value, Tensor):
                log.debug("Cannot log item which is not Tensor")
                continue
            if value.numel() != 1:
                log.debug("Cannot log Tensor which is not scalar")
                continue

            self.log(
                f"{subset_name}/{metric}",
                value,
                sync_dist=True,
                prog_bar=True,
            )

    def training_step(
        self,
        inputs: VisualPromptingBatchDataEntity | ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
        batch_idx: int,
    ) -> Tensor:
        """Step for model training."""
        train_loss = self.model(inputs)

        if isinstance(train_loss, Tensor):
            self.train_metric["Loss"].update(train_loss)

        elif isinstance(train_loss, dict):
            for k, v in train_loss.items():
                if k in self.train_metric:
                    self.train_metric[k].update(v)

        else:
            raise TypeError(train_loss)

        self._log_metrics(self.train_metric, "train")

        return train_loss

    def validation_step(self, inputs: VisualPromptingBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            inputs (VisualPromptingBatchDataEntity): The input data for the validation step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type VisualPromptingBatchPredEntity.

        Returns:
            None
        """
        self._inference_step(self.val_metric, inputs, batch_idx)

    def test_step(self, inputs: VisualPromptingBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            inputs (VisualPromptingBatchDataEntity): The input data for the test step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type VisualPromptingBatchPredEntity.
        """
        self._inference_step(self.test_metric, inputs, batch_idx)

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: VisualPromptingBatchPredEntity | ZeroShotVisualPromptingBatchPredEntity,
        inputs: VisualPromptingBatchDataEntity | ZeroShotVisualPromptingBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        """Convert the prediction entity to the format required by the compute metric function."""
        pred_info = []
        target_info = []

        for masks, scores, labels in zip(
            preds.masks,
            preds.scores,
            preds.labels,
        ):
            pred_info.append(
                {
                    "masks": masks.data,
                    "scores": scores,
                    "labels": labels,
                },
            )

        for imgs_info, masks, polygons, labels in zip(
            inputs.imgs_info,
            inputs.masks,
            inputs.polygons,
            inputs.labels,
        ):
            bit_masks = masks if len(masks) else polygon_to_bitmap(polygons, *imgs_info.ori_shape)
            target_info.append(
                {
                    "masks": tv_tensors.Mask(bit_masks, dtype=torch.bool).data,
                    "labels": labels,
                },
            )

        return {"preds": pred_info, "target": target_info}

    def _inference_step(
        self,
        metric: MetricCollection,
        inputs: VisualPromptingBatchDataEntity | ZeroShotVisualPromptingBatchDataEntity,
        batch_idx: int,
    ) -> None:
        """Perform a single inference step on a batch of data from the inference set."""
        preds = self.model(inputs)

        if not isinstance(preds, VisualPromptingBatchPredEntity):
            raise TypeError(preds)

        converted_entities = self._convert_pred_entity_to_compute_metric(preds, inputs)
        for _name, _metric in metric.items():
            if _name == "mAP":
                # MeanAveragePrecision
                _preds = [
                    {k: v > 0.5 if k == "masks" else v.squeeze(1) if k == "scores" else v for k, v in ett.items()}
                    for ett in converted_entities["preds"]
                ]
                _target = converted_entities["target"]
                _metric.update(preds=_preds, target=_target)
            elif _name in ["IoU", "F1", "Dice"]:
                # BinaryJaccardIndex, BinaryF1Score, Dice
                for cvt_preds, cvt_target in zip(converted_entities["preds"], converted_entities["target"]):
                    _metric.update(cvt_preds["masks"], cvt_target["masks"])


class OTXZeroShotVisualPromptingLitModule(OTXVisualPromptingLitModule):
    """Base class for the lightning module used in OTX zero-shot visual prompting task."""

    def set_metrics(self) -> None:
        """Set metrics."""
        self.test_metric = MetricCollection(
            {
                "IoU": BinaryJaccardIndex(),
                "F1": BinaryF1Score(),
                "Dice": Dice(),
                "mAP": MeanAveragePrecision(iou_type="segm"),
            },
        )

    def on_train_epoch_start(self) -> None:
        """Skip on_train_epoch_start unused in zero-shot visual prompting."""

    def on_train_epoch_end(self) -> None:
        """Skip on_train_epoch_end unused in zero-shot visual prompting."""

    def on_validation_epoch_start(self) -> None:
        """Skip on_validation_epoch_start unused in zero-shot visual prompting."""

    def on_validation_epoch_end(self) -> None:
        """Skip on_validation_epoch_end unused in zero-shot visual prompting."""

    def configure_optimizers(self) -> None:  # type: ignore[override]
        """Skip configure_optimizers unused in zero-shot visual prompting."""

    def training_step(
        self,
        inputs: VisualPromptingBatchDataEntity | ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
        batch_idx: int,
    ) -> Tensor:
        """Skip training_step unused in zero-shot visual prompting."""
        self.model(inputs)

    def validation_step(
        self,
        inputs: VisualPromptingBatchDataEntity | ZeroShotVisualPromptingBatchDataEntity,
        batch_idx: int,
    ) -> None:
        """Skip validation_step unused in zero-shot visual prompting."""

    def _inference_step(
        self,
        metric: MetricCollection,
        inputs: VisualPromptingBatchDataEntity | ZeroShotVisualPromptingBatchDataEntity,
        batch_idx: int,
    ) -> None:
        """Perform a single inference step on a batch of data from the inference set."""
        preds = self.model(inputs)

        if not isinstance(preds, ZeroShotVisualPromptingBatchPredEntity):
            raise TypeError(preds)

        converted_entities = self._convert_pred_entity_to_compute_metric(preds, inputs)
        for _name, _metric in metric.items():
            if _name == "mAP":
                # MeanAveragePrecision
                _preds = [
                    {
                        k: v > 0.5 if k == "masks" else v.squeeze(1).to(self.device) if k == "labels" else v
                        for k, v in ett.items()
                    }
                    for ett in converted_entities["preds"]
                ]
                _target = converted_entities["target"]

                # match #_preds and #_target
                if len(_preds) > len(_target):
                    # interpolate _target
                    num_diff = len(_preds) - len(_target)
                    for idx in range(num_diff):
                        _target.append(_target[idx])
                elif len(_preds) < len(_target):
                    num_diff = len(_target) - len(_preds)
                    pad_prediction = {
                        "masks": torch.zeros_like(_target[0]["masks"], dtype=_target[0]["masks"].dtype),
                        "labels": torch.zeros_like(_target[0]["labels"], dtype=_target[0]["labels"].dtype),
                        "scores": torch.zeros(len(_target[0]["labels"]), dtype=torch.float32),
                    }  # for empty prediction
                    for idx in range(num_diff):
                        _preds.append(_preds[idx] if idx < len(_preds) else pad_prediction)

                _metric.update(preds=_preds, target=_target)
            elif _name in ["IoU", "F1", "Dice"]:
                # BinaryJaccardIndex, BinaryF1Score, Dice
                for cvt_preds, cvt_target in zip(converted_entities["preds"], converted_entities["target"]):
                    _metric.update(
                        cvt_preds["masks"].sum(dim=0).clamp(0, 1),
                        cvt_target["masks"].sum(dim=0).clamp(0, 1),
                    )
