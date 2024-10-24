# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for image captioning model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from otx.core.data.entity.image_captioning import ImageCaptionBatchDataEntity, ImageCaptionBatchPredEntity
from otx.core.metrics import MetricCallable
from otx.core.metrics.clip_score import CLIPScoreCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes
from otx.core.types.task import OTXTrainType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable


class ImageCaptioningModel(OTXModel[ImageCaptionBatchDataEntity, ImageCaptionBatchPredEntity]):
    """Base class for the image captioning models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = CLIPScoreCallable,
        torch_compile: bool = False,
        train_type: Literal[OTXTrainType.SUPERVISED] = OTXTrainType.SUPERVISED,
    ) -> None:
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            train_type=train_type,
        )
