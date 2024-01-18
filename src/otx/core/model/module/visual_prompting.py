# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting lightning module used in OTX."""
from __future__ import annotations

import logging as log

import torch
from torch import Tensor
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, Dice
from torchmetrics.collections import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors

from otx.core.data.entity.visual_prompting import VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity
from otx.core.model.entity.visual_prompting import OTXVisualPromptingModel
from otx.core.model.module.base import OTXLitModule
from otx.core.utils.mask_util import polygon_to_bitmap


class OTXVisualPromptingLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX visual prompting task."""

    def __init__(
        self,
        otx_model: OTXVisualPromptingModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        torch_compile: bool,
    ):
        super().__init__(otx_model, optimizer, scheduler, torch_compile)

        self.train_metric = MetricCollection(
            {
                "loss": MeanMetric(),
                "loss_dice": MeanMetric(),
                "loss_focal": MeanMetric(),
                "loss_iou": MeanMetric(),
            },
        )
        self.val_metric = MetricCollection(
            {
                "IoU": BinaryJaccardIndex(),
                "F1": BinaryF1Score(),
                "Dice": Dice(),
                "mAP": MeanAveragePrecision(iou_type="segm"),
            },
        )
        self.test_metric = MetricCollection(
            {
                "IoU": BinaryJaccardIndex(),
                "F1": BinaryF1Score(),
                "Dice": Dice(),
                "mAP": MeanAveragePrecision(iou_type="segm"),
            },
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

    def training_step(self, inputs: VisualPromptingBatchDataEntity, batch_idx: int) -> Tensor:  # type: ignore[override]
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
        preds = self.model(inputs)

        if not isinstance(preds, VisualPromptingBatchPredEntity):
            raise TypeError(preds)

        converted_entities = self._convert_pred_entity_to_compute_metric(preds, inputs)
        for name, metric in self.val_metric.items():
            if name == "mAP":
                # MeanAveragePrecision
                _preds = [
                    {k: v > 0.5 if k == "masks" else v.squeeze(1) if k == "scores" else v for k, v in ett.items()}
                    for ett in converted_entities["preds"]
                ]
                _target = converted_entities["target"]
                metric.update(preds=_preds, target=_target)
            elif name in ["IoU", "F1", "Dice"]:
                # BinaryJaccardIndex, BinaryF1Score, Dice
                for cvt_preds, cvt_target in zip(converted_entities["preds"], converted_entities["target"]):
                    metric.update(cvt_preds["masks"], cvt_target["masks"])

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: VisualPromptingBatchPredEntity,
        inputs: VisualPromptingBatchDataEntity,
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

    def test_step(self, inputs: VisualPromptingBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            inputs (VisualPromptingBatchDataEntity): The input data for the test step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type VisualPromptingBatchPredEntity.
        """
        preds = self.model(inputs)

        if not isinstance(preds, VisualPromptingBatchPredEntity):
            raise TypeError(preds)

        converted_entities = self._convert_pred_entity_to_compute_metric(preds, inputs)
        for name, metric in self.test_metric.items():
            if name == "mAP":
                # MeanAveragePrecision
                _preds = [
                    {k: v > 0.5 if k == "masks" else v.squeeze(1) if k == "scores" else v for k, v in ett.items()}
                    for ett in converted_entities["preds"]
                ]
                _target = converted_entities["target"]
                metric.update(preds=_preds, target=_target)
            elif name in ["IoU", "F1", "Dice"]:
                # BinaryJaccardIndex, BinaryF1Score, Dice
                for cvt_preds, cvt_target in zip(converted_entities["preds"], converted_entities["target"]):
                    metric.update(cvt_preds["masks"], cvt_target["masks"])

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"
