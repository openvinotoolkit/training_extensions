# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection lightning module used in OTX."""
from __future__ import annotations

import logging as log

import torch
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors

from otx.core.data.entity.instance_segmentation import (
    InstanceSegBatchDataEntity,
    InstanceSegBatchPredEntity,
)
from otx.core.model.entity.instance_segmentation import OTXInstanceSegModel
from otx.core.model.module.base import OTXLitModule
from otx.core.utils.mask_util import polygon_to_bitmap


class OTXInstanceSegLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX detection task."""

    def __init__(
        self,
        otx_model: OTXInstanceSegModel,
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

    def validation_step(self, inputs: InstanceSegBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, InstanceSegBatchPredEntity):
            raise TypeError(preds)

        self.val_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: InstanceSegBatchPredEntity,
        inputs: InstanceSegBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        """Convert the prediction entity to the format required by the compute metric function."""
        pred_info = []
        target_info = []

        for bboxes, masks, scores, labels in zip(
            preds.bboxes, preds.masks, preds.scores, preds.labels,
        ):
            pred_info.append({
                "boxes": bboxes.data,
                "masks": masks.data,
                "scores": scores,
                "labels": labels,
            })

        for imgs_info, bboxes, masks, polygons, labels in zip(
            inputs.imgs_info,
            inputs.bboxes,
            inputs.masks,
            inputs.polygons,
            inputs.labels,
        ):
            bit_masks = masks if len(masks) else polygon_to_bitmap(polygons, *imgs_info.ori_shape)
            target_info.append({
                "boxes": bboxes.data,
                "masks": tv_tensors.Mask(bit_masks, dtype=torch.bool).data,
                "labels": labels,
            })

        return {"preds": pred_info, "target": target_info}

    def test_step(self, inputs: InstanceSegBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, InstanceSegBatchPredEntity):
            raise TypeError(preds)

        self.test_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"
