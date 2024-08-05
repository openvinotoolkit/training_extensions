# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging-Face pretrained model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForImageClassification

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    CLASSIFICATION_BATCH_DATA_ENTITY,
    MulticlassClsBatchPredEntity,
)
from otx.core.metrics.accuracy import DefaultClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import OTXClassificationModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes
from otx.core.types.task import OTXTaskType, OTXTrainType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from transformers.modeling_outputs import ImageClassifierOutput

    from otx.core.metrics import MetricCallablePerTask


class HuggingFaceModelForMulticlassCls(OTXClassificationModel):
    """HuggingFaceModelForMulticlassCls is a class that represents a Hugging Face model for multiclass classification.

    Args:
        model_name_or_path (str): The name or path of the pretrained model. You can find available models at https://huggingface.co/models?pipeline_tag=image-classification.
        label_info (LabelInfoTypes): The label information for the classification task.
        optimizer (OptimizerCallable, optional): The optimizer callable for training the model.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

    Example:
        1. API
            >>> model = HuggingFaceModelForMulticlassCls(
            ...     model_name_or_path="facebook/deit-tiny-patch16-224",
            ...     label_info=<Number-of-classes>,
            ... )
        2. CLI
            >>> otx train \
            ... --model otx.algo.classification.huggingface_model.HuggingFaceModelForMulticlassCls \
            ... --model.model_name_or_path facebook/deit-tiny-patch16-224
    """

    def __init__(
        self,
        model_name_or_path: str,  # https://huggingface.co/models?pipeline_tag=image-classification
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallablePerTask = DefaultClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.model_name = model_name_or_path

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            task=OTXTaskType.MULTI_CLASS_CLS,
            train_type=OTXTrainType.SUPERVISED,
        )

    def _create_model(self) -> nn.Module:
        return AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=self.label_info.num_classes,
            ignore_mismatched_sizes=True,
        )

    def _customize_inputs(self, inputs: CLASSIFICATION_BATCH_DATA_ENTITY) -> dict[str, Any]:
        return {
            "pixel_values": inputs.images,
            "labels": torch.cat(inputs.labels, dim=0),
        }

    def _customize_outputs(
        self,
        outputs: ImageClassifierOutput,
        inputs: CLASSIFICATION_BATCH_DATA_ENTITY,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs.loss)

        logits = outputs.logits
        scores = torch.unbind(logits, 0)
        preds = logits.argmax(-1, keepdim=True).unbind(0)

        return MulticlassClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=preds,
        )

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            msg = "Explain mode is not supported for this model."
            raise NotImplementedError(msg)

        return self.model(pixel_values=image)
