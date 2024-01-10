# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting lightning module used in OTX."""
from __future__ import annotations

import logging as log

import torch
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors

from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity)
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

        self.val_metric = MeanAveragePrecision(iou_type="segm")
        self.test_metric = MeanAveragePrecision(iou_type="segm")

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

    def _log_metrics(self, meter: MeanAveragePrecision, subset_name: str) -> None:
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

        self.val_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: VisualPromptingBatchPredEntity,
        inputs: VisualPromptingBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        """Convert the prediction entity to the format required by the compute metric function."""
        pred_info = []
        target_info = []

        for bboxes, masks, scores, labels in zip(
            preds.bboxes,
            preds.masks,
            preds.scores,
            preds.labels,
        ):
            pred_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": masks.data,
                    "scores": scores,
                    "labels": labels,
                },
            )

        for imgs_info, bboxes, masks, polygons, labels in zip(
            inputs.imgs_info,
            inputs.bboxes,
            inputs.masks,
            inputs.polygons,
            inputs.labels,
        ):
            bit_masks = masks if len(masks) else polygon_to_bitmap(polygons, *imgs_info.ori_shape)
            target_info.append(
                {
                    "boxes": bboxes.data,
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

        self.test_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"