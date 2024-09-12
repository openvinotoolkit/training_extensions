# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Anomaly Lightning OTX model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from anomalib import TaskType as AnomalibTaskType
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization.min_max_normalization import _MinMaxNormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from torch import nn

from otx.core.data.entity.anomaly import (
    AnomalyClassificationBatchPrediction,
    AnomalyClassificationDataBatch,
    AnomalyDetectionBatchPrediction,
    AnomalyDetectionDataBatch,
    AnomalySegmentationBatchPrediction,
    AnomalySegmentationDataBatch,
)
from otx.core.data.entity.base import ImageInfo
from otx.core.exporter.anomaly import OTXAnomalyModelExporter
from otx.core.model.base import OTXModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from pathlib import Path

    from anomalib.metrics import AnomalibMetricCollection
    from anomalib.metrics.threshold import BaseThreshold
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks.callback import Callback
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torchmetrics import Metric

    from otx.core.types.label import LabelInfoTypes

AnomalyModelInputs: TypeAlias = (
    AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch
)
AnomalyModelOutputs: TypeAlias = (
    AnomalyClassificationBatchPrediction | AnomalySegmentationBatchPrediction | AnomalyDetectionBatchPrediction
)


class OTXAnomaly(OTXModel):
    """Methods used to make OTX model compatible with the Anomalib model.

    Args:
        input_size (tuple[int, int] | None):
            Model input size in the order of height and width. Defaults to None.
    """

    def __init__(self, label_info: LabelInfoTypes, input_size: tuple[int, int]) -> None:
        super().__init__(label_info=label_info, input_size=input_size)
        self.optimizer: list[OptimizerCallable] | OptimizerCallable = None
        self.scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = None
        self.trainer: Trainer
        self.model: nn.Module
        self.image_threshold: BaseThreshold
        self.pixel_threshold: BaseThreshold

        self.normalization_metrics: Metric

        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on saving checkpoint."""
        super().on_save_checkpoint(checkpoint)  # type: ignore[misc]

        attrs = ["_task_type", "_input_size", "image_threshold", "pixel_threshold"]
        checkpoint["anomaly"] = {key: getattr(self, key, None) for key in attrs}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on loading checkpoint."""
        super().on_load_checkpoint(checkpoint)  # type: ignore[misc]
        if anomaly_attrs := checkpoint.get("anomaly"):
            for key, value in anomaly_attrs.items():
                setattr(self, key, value)

    @property  # type: ignore[override]
    def input_size(self) -> tuple[int, int]:
        """Returns the input size of the model.

        Returns:
            tuple[int, int]: The input size of the model as a tuple of (height, width).
        """
        return self._input_size

    @input_size.setter
    def input_size(self, value: tuple[int, int]) -> None:
        self._input_size = value

    @property
    def task(self) -> AnomalibTaskType:
        """Return the task type of the model."""
        if self._task_type:
            return self._task_type
        msg = "``self._task_type`` is not assigned"
        raise AttributeError(msg)

    @task.setter
    def task(self, value: OTXTaskType) -> None:
        if value == OTXTaskType.ANOMALY_CLASSIFICATION:
            self._task_type = AnomalibTaskType.CLASSIFICATION
        elif value == OTXTaskType.ANOMALY_DETECTION:
            self._task_type = AnomalibTaskType.DETECTION
        elif value == OTXTaskType.ANOMALY_SEGMENTATION:
            self._task_type = AnomalibTaskType.SEGMENTATION
        else:
            msg = f"Unexpected task type: {value}"
            raise ValueError(msg)

    def _get_values_from_transforms(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get the value requested value from default transforms."""
        mean_value, std_value = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
        for transform in self.configure_transforms().transforms:  # type: ignore[attr-defined]
            name = transform.__class__.__name__
            if "Normalize" in name:
                mean_value = tuple(value * 255 for value in transform.mean)  # type: ignore[assignment]
                std_value = tuple(value * 255 for value in transform.std)  # type: ignore[assignment]
        return mean_value, std_value

    @property
    def trainable_model(self) -> str | None:
        """Use this to return the name of the model that needs to be trained.

        This might not be the cleanest solution.

        Some models have multiple architectures and only one of them needs to be trained.
        However the optimizer is configured in the Anomalib's lightning model. This can be used
        to inform the OTX lightning model which model to train.
        """
        return None

    def configure_callbacks(self) -> list[Callback]:
        """Get all necessary callbacks required for training and post-processing on Anomalib models."""
        image_metrics = ["AUROC", "F1Score"]
        pixel_metrics = image_metrics if self.task != AnomalibTaskType.CLASSIFICATION else None
        return [
            _PostProcessorCallback(),
            _MinMaxNormalizationCallback(),  # ModelAPI only supports min-max normalization as of now
            _ThresholdCallback(threshold="F1AdaptiveThreshold"),
            _MetricsCallback(
                task=self.task,
                image_metrics=image_metrics,
                pixel_metrics=pixel_metrics,
            ),
        ]

    def on_predict_batch_end(
        self,
        outputs: dict,
        batch: AnomalyModelInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Wrap the outputs to OTX format.

        Since outputs need to be replaced inplace, we can't change the datatype of outputs.
        That's why outputs is cleared and replaced with the new outputs. The problem with this is that
        Instead of ``engine.test()`` returning [BatchPrediction,...], it returns
        [{prediction: BatchPrediction}, {...}, ...]
        """
        _outputs = self._customize_outputs(outputs, batch)
        outputs.clear()
        outputs.update({"prediction": _outputs})

    def _customize_inputs(
        self,
        inputs: AnomalyModelInputs,
    ) -> dict[str, Any]:
        """Customize inputs for the model."""
        return_dict = {"image": inputs.images, "label": torch.vstack(inputs.labels).squeeze()}
        if isinstance(inputs, AnomalySegmentationDataBatch) and inputs.masks is not None:
            return_dict["mask"] = inputs.masks
        if isinstance(inputs, AnomalyDetectionDataBatch) and inputs.masks is not None and inputs.boxes is not None:
            return_dict["mask"] = inputs.masks
            return_dict["boxes"] = inputs.boxes

        if return_dict["label"].size() == torch.Size([]):  # when last batch size is 1
            return_dict["label"] = return_dict["label"].unsqueeze(0)
        return return_dict

    def _customize_outputs(
        self,
        outputs: dict,
        inputs: AnomalyModelInputs,
    ) -> AnomalyModelOutputs:
        if self.task == AnomalibTaskType.CLASSIFICATION:
            return AnomalyClassificationBatchPrediction(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                labels=outputs["label"],
                # Note: this is the anomalous score. It should be inverted to report Normal score
                scores=outputs["pred_scores"],
                anomaly_maps=outputs["anomaly_maps"],
            )
        if self.task == AnomalibTaskType.SEGMENTATION:
            return AnomalySegmentationBatchPrediction(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                labels=outputs["label"],
                # Note: this is the anomalous score. It should be inverted to report Normal score
                scores=outputs["pred_scores"],
                anomaly_maps=outputs["anomaly_maps"],
                masks=outputs["mask"],
            )
        if self.task == AnomalibTaskType.DETECTION:
            return AnomalyDetectionBatchPrediction(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                labels=outputs["label"],
                # Note: this is the anomalous score. It should be inverted to report Normal score
                scores=outputs["pred_scores"],
                anomaly_maps=outputs["anomaly_maps"],
                masks=outputs["mask"],
                boxes=outputs["pred_boxes"],
                box_scores=outputs["box_scores"],
                box_labels=outputs["box_labels"],
            )
        msg = f"Unsupported task type {self.task}"
        raise ValueError(msg)

    @property
    def _exporter(self) -> OTXAnomalyModelExporter:
        """Creates OTXAnomalyModelExporter object that can export anomaly models."""
        min_val = self.normalization_metrics.state_dict()["min"].cpu().numpy().tolist()
        max_val = self.normalization_metrics.state_dict()["max"].cpu().numpy().tolist()
        mean_values, scale_values = self._get_values_from_transforms()
        onnx_export_configuration = {
            "opset_version": 14,
            "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            "input_names": ["input"],
            "output_names": ["output"],
        }
        return OTXAnomalyModelExporter(
            image_shape=self.input_size,
            image_threshold=self.image_threshold.value.cpu().numpy().tolist(),
            pixel_threshold=self.pixel_threshold.value.cpu().numpy().tolist(),
            task=self.task,
            mean_values=mean_values,
            scale_values=scale_values,
            normalization_scale=max_val - min_val,
            onnx_export_configuration=onnx_export_configuration,
            via_onnx=False,
        )

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        to_exportable_code: bool = False,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model
            to_exportable_code (bool): flag to export model in exportable code with demo package

        Returns:
            Path: path to the exported model.
        """
        return self._exporter.export(
            model=self.model,
            output_dir=output_dir,
            base_model_name=base_name,
            export_format=export_format,
            precision=precision,
            to_exportable_code=to_exportable_code,
        )

    def get_dummy_input(self, batch_size: int = 1) -> AnomalyModelInputs:
        """Returns a dummy input for anomaly model."""
        images = torch.rand(batch_size, 3, *self.input_size)
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        if self.task == AnomalibTaskType.CLASSIFICATION:
            return AnomalyClassificationDataBatch(
                batch_size=batch_size,
                images=images,
                imgs_info=infos,
                labels=[torch.LongTensor(0)],
            )
        if self.task == AnomalibTaskType.SEGMENTATION:
            return AnomalySegmentationDataBatch(
                batch_size=batch_size,
                images=images,
                imgs_info=infos,
                labels=[torch.LongTensor(0)],
                masks=torch.tensor(0),
            )
        if self.task == AnomalibTaskType.DETECTION:
            return AnomalyDetectionDataBatch(
                batch_size=batch_size,
                images=images,
                imgs_info=infos,
                labels=[torch.LongTensor(0)],
                boxes=torch.tensor(0),
                masks=torch.tensor(0),
            )

        msg = "Wrong anomaly task type"
        raise RuntimeError(msg)
