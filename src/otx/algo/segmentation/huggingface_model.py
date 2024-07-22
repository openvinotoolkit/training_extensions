# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging-Face pretrained model for the OTX Semantic Segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.dice import SegmCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.segmentation import OTXSegmentationModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from transformers.modeling_outputs import SemanticSegmenterOutput

    from otx.core.metrics import MetricCallable


class HuggingFaceModelForSegmentation(OTXSegmentationModel):
    """A class representing a Hugging Face model for segmentation.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model.
        label_info (LabelInfoTypes): The label information for the model.
        optimizer (OptimizerCallable, optional): The optimizer for training the model.
            Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional):
            The learning rate scheduler for training the model. Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): The evaluation metric for the model.
            Defaults to MeanAveragePrecisionFMeasureCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

    Example:
        1. API
            >>> model = HuggingFaceModelForSegmentation(
            ...     model_name_or_path="nvidia/segformer-b0-finetuned-ade-512-512",
            ...     label_info=<Number-of-classes>,
            ... )
        2. CLI
            >>> otx train \
            ... --model otx.algo.segmentation.huggingface_model.HuggingFaceModelForSegmentation \
            ... --model.model_name_or_path nvidia/segformer-b0-finetuned-ade-512-512
    """

    def __init__(
        self,
        model_name_or_path: str,  # https://huggingface.co/models?pipeline_tag=image-segmentation
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
    ) -> None:
        self.model_name = model_name_or_path
        self.load_from = None

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)

    def _create_model(self) -> nn.Module:
        return AutoModelForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=self.label_info.num_classes,
            ignore_mismatched_sizes=True,
        )

    def _customize_inputs(self, entity: SegBatchDataEntity) -> dict[str, Any]:
        masks = torch.stack(entity.masks).long() if self.training else None

        return {
            "pixel_values": entity.images,
            "labels": masks,
        }

    def _customize_outputs(
        self,
        outputs: SemanticSegmenterOutput,
        inputs: SegBatchDataEntity,
    ) -> SegBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs.loss)
        if self.explain_mode:
            msg = "Explain mode is not supported yet."
            raise NotImplementedError(msg)

        target_sizes = [info.img_shape for info in inputs.imgs_info]
        results = self.image_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_sizes,
        )

        return SegBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=results,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        size = self.image_processor.size.values()
        size = (*size, *size) if len(size) == 1 else size
        image_size = (1, 3, *size)
        image_mean = (123.675, 116.28, 103.53)
        image_std = (58.395, 57.12, 57.375)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=image_size,
            mean=image_mean,
            std=image_std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=None,
        )

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model(image)
