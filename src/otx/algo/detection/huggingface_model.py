# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging-Face pretrained model for the OTX Object Detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torchvision import tv_tensors
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import OTXDetectionModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from transformers.utils.generic import ModelOutput

    from otx.core.metrics import MetricCallable


class HuggingFaceModelForDetection(OTXDetectionModel):
    """A class representing a Hugging Face model for object detection.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model.
        label_info (LabelInfoTypes): The label information for the model.
        input_size (tuple[int, int], optional):
            Model input size in the order of height and width. Defaults to (800, 992).
        optimizer (OptimizerCallable, optional): The optimizer for training the model.
            Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional):
            The learning rate scheduler for training the model. Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): The evaluation metric for the model.
            Defaults to MeanAveragePrecisionFMeasureCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

    Example:
        1. API
            >>> model = HuggingFaceModelForDetection(
            ...     model_name_or_path="facebook/detr-resnet-50",
            ...     label_info=<Number-of-classes>,
            ... )
        2. CLI
            >>> otx train \
            ... --model otx.algo.detection.huggingface_model.HuggingFaceModelForDetection \
            ... --model.model_name_or_path facebook/detr-resnet-50
    """

    def __init__(
        self,
        model_name_or_path: str,  # https://huggingface.co/models?pipeline_tag=object-detection
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (800, 992),  # input size of default detection data recipe
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
    ) -> None:
        self.model_name = model_name_or_path
        self.load_from = None

        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)

    def _build_model(self, num_classes: int) -> nn.Module:
        return AutoModelForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def _customize_inputs(
        self,
        entity: DetBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        labels = [{"class_labels": entity.labels[i], "boxes": entity.bboxes[i]} for i in range(entity.batch_size)]

        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(
                entity.images,
                entity.imgs_info,
                pad_size_divisor=pad_size_divisor,
                pad_value=pad_value,
            )

        return {
            "pixel_values": entity.images,
            "labels": labels,
        }

    def _customize_outputs(
        self,
        outputs: ModelOutput,
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return outputs.loss_dict

        target_sizes = torch.tensor([box.canvas_size for box in inputs.bboxes])
        results = self.image_processor.post_process_object_detection(
            outputs,
            0.0,
            target_sizes=target_sizes,
        )

        scores, labels, bboxes = [], [], []
        for i, result in enumerate(results):
            scores.append(result["scores"])
            labels.append(result["labels"])
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    result["boxes"],
                    format="XYXY",
                    canvas_size=inputs.bboxes[i].canvas_size,
                ),
            )

        if self.explain_mode:
            msg = "Explain mode is not supported yet."
            raise NotImplementedError(msg)

        return DetBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
            bboxes=bboxes,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        image_mean = (0.0, 0.0, 0.0)
        image_std = (255.0, 255.0, 255.0)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=image_mean,  # type: ignore[arg-type]
            std=image_std,  # type: ignore[arg-type]
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "autograd_inlining": False,
                "opset_version": 16,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(self, inputs: torch.Tensor) -> dict[str, Any]:  # type: ignore[override]
        """Forward function for export."""
        outputs = self.model(inputs)
        results = self.image_processor.post_process_object_detection(
            outputs,
            0.0,
        )
        scores, labels, bboxes = [], [], []
        for result in results:
            scores.append(result["scores"])
            labels.append(result["labels"])
            bboxes.append(result["boxes"])

        return {
            "bboxes": torch.stack(bboxes),
            "labels": torch.stack(labels),
            "scores": torch.stack(scores),
        }
