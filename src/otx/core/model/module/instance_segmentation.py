# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for instance segmentation lightning module used in OTX."""
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from otx.algo.instance_segmentation.otx_instseg_evaluation import (
    OTXInstSegMeanAveragePrecision,
)
from otx.core.data.entity.instance_segmentation import (
    InstanceSegBatchDataEntity,
    InstanceSegBatchPredEntity,
)
from otx.core.model.entity.instance_segmentation import OTXInstanceSegModel
from otx.core.model.module.base import OTXLitModule
from otx.core.utils.mask_util import encode_rle, polygon_to_rle

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable


class OTXInstanceSegLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX instance segmentation task."""

    def __init__(
        self,
        otx_model: OTXInstanceSegModel,
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

        self.val_metric = OTXInstSegMeanAveragePrecision(iou_type="segm")
        self.test_metric = OTXInstSegMeanAveragePrecision(iou_type="segm")

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

    def _log_metrics(self, meter: OTXInstSegMeanAveragePrecision, subset_name: str) -> None:
        results = meter.compute()
        if results is None:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

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

        Args:
            inputs (InstanceSegBatchDataEntity): The input data for the validation step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type InstanceSegBatchPredEntity.

        Returns:
            None
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
        """Convert the prediction entity to the format that the metric can compute and cache the ground truth.

        This function will convert mask to RLE format and cache the ground truth for the current batch.

        Args:
            preds (InstanceSegBatchPredEntity): Current batch predictions.
            inputs (InstanceSegBatchDataEntity): Current batch ground-truth inputs.

        Returns:
            dict[str, list[dict[str, Tensor]]]: The converted predictions and ground truth.
        """
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
                    "masks": [encode_rle(mask) for mask in masks.data],
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
            rles = (
                [encode_rle(mask) for mask in masks.data]
                if len(masks)
                else polygon_to_rle(polygons, *imgs_info.ori_shape)
            )
            target_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": rles,
                    "labels": labels,
                },
            )
        return {"preds": pred_info, "target": target_info}

    def test_step(self, inputs: InstanceSegBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            inputs (InstanceSegBatchDataEntity): The input data for the test step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type InstanceSegBatchPredEntity.
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
        return "val/map_50"
