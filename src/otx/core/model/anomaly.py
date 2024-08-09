# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Anomaly Lightning OTX model."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
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
from otx.core.types.label import AnomalyLabelInfo
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from pathlib import Path

    from anomalib.metrics import AnomalibMetricCollection
    from anomalib.metrics.threshold import BaseThreshold
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks.callback import Callback
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from torch.optim.optimizer import Optimizer
    from torchmetrics import Metric

AnomalyModelInputs: TypeAlias = (
    AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch
)
AnomalyModelOutputs: TypeAlias = (
    AnomalyClassificationBatchPrediction | AnomalySegmentationBatchPrediction | AnomalyDetectionBatchPrediction
)


class OTXAnomaly(OTXModel):
    """Methods used to make OTX model compatible with the Anomalib model."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(label_info=AnomalyLabelInfo())
        self.optimizer: list[OptimizerCallable] | OptimizerCallable = None
        self.scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = None
        self._input_size: tuple[int, int] = (256, 256)
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

    @property
    def input_size(self) -> tuple[int, int]:
        """Returns the input size of the model.

        Returns:
            tuple[int, int]: The input size of the model as a tuple of (height, width).
        """
        return self._input_size

    @input_size.setter
    def input_size(self, value: tuple[int, int]) -> None:
        self._input_size = value

    def _get_values_from_transforms(
        self,
    ) -> tuple[tuple[int, int], tuple[float, float, float], tuple[float, float, float]]:
        """Get the value requested value from default transforms."""
        image_size, mean_value, std_value = (256, 256), (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
        for transform in self.configure_transforms().transforms:  # type: ignore[attr-defined]
            name = transform.__class__.__name__
            if "Resize" in name:
                image_size = tuple(transform.size)  # type: ignore[assignment]
            elif "Normalize" in name:
                mean_value = tuple(value * 255 for value in transform.mean)  # type: ignore[assignment]
                std_value = tuple(value * 255 for value in transform.std)  # type: ignore[assignment]
        return image_size, mean_value, std_value

    def configure_metric(self) -> None:
        """This does not follow OTX metric configuration."""
        return

    def configure_callbacks(self) -> list[Callback]:
        """Get all necessary callbacks required for training and post-processing on Anomalib models."""
        image_metrics = ["AUROC", "F1Score"]
        pixel_metrics = image_metrics if self.task != OTXTaskType.ANOMALY_CLASSIFICATION else None
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

    def on_validation_epoch_start(self) -> None:
        """Don't call OTXModel's ``on_validation_epoch_start``."""
        return

    def on_test_epoch_start(self) -> None:
        """Don't call OTXModel's ``on_test_epoch_start``."""
        return

    def on_validation_epoch_end(self) -> None:
        """Don't call OTXModel's ``on_validation_epoch_end``."""
        return

    def on_test_epoch_end(self) -> None:
        """Don't call OTXModel's ``on_test_epoch_end``."""
        return

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
        return_dict = {}
        if isinstance(inputs, AnomalyClassificationDataBatch):
            return_dict = {"image": inputs.images, "label": torch.vstack(inputs.labels).squeeze()}
        if isinstance(inputs, AnomalySegmentationDataBatch):
            return_dict = {"image": inputs.images, "label": torch.vstack(inputs.labels).squeeze(), "mask": inputs.masks}
        if isinstance(inputs, AnomalyDetectionDataBatch):
            return_dict = {
                "image": inputs.images,
                "label": torch.vstack(inputs.labels).squeeze(),
                "mask": inputs.masks,
                "boxes": inputs.boxes,
            }

        if return_dict["label"].size() == torch.Size([]):  # when last batch size is 1
            return_dict["label"] = return_dict["label"].unsqueeze(0)
        return return_dict

    def _customize_outputs(
        self,
        outputs: dict,
        inputs: AnomalyModelInputs,
    ) -> AnomalyModelOutputs:
        if self.task == OTXTaskType.ANOMALY_CLASSIFICATION:
            return AnomalyClassificationBatchPrediction(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                labels=outputs["label"],
                # Note: this is the anomalous score. It should be inverted to report Normal score
                scores=outputs["pred_scores"],
                anomaly_maps=outputs["anomaly_maps"],
            )
        if self.task == OTXTaskType.ANOMALY_SEGMENTATION:
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
        if self.task == OTXTaskType.ANOMALY_DETECTION:
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
        image_shape, mean_values, scale_values = self._get_values_from_transforms()
        onnx_export_configuration = {
            "opset_version": 14,
            "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            "input_names": ["input"],
            "output_names": ["output"],
        }
        return OTXAnomalyModelExporter(
            image_shape=image_shape,
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
        if export_format == OTXExportFormatType.OPENVINO:
            if to_exportable_code:
                msg = "Exportable code option is not supported yet for anomaly tasks and will be ignored."
                log.warning(msg)
            to_exportable_code = False

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
        image_size, _, _ = self._get_values_from_transforms()
        images = torch.rand(batch_size, 3, *image_size)
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        if self.task == OTXTaskType.ANOMALY_CLASSIFICATION:
            return AnomalyClassificationDataBatch(
                batch_size=batch_size,
                images=images,
                imgs_info=infos,
                labels=[torch.LongTensor(0)],
            )
        if self.task == OTXTaskType.ANOMALY_SEGMENTATION:
            return AnomalySegmentationDataBatch(
                batch_size=batch_size,
                images=images,
                imgs_info=infos,
                labels=[torch.LongTensor(0)],
                masks=torch.tensor(0),
            )
        if self.task == OTXTaskType.ANOMALY_DETECTION:
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


class AnomalyMixin:
    """Mixin inherited before AnomalibModule to override OTXModel methods."""

    def configure_optimizers(self) -> tuple[list[Optimizer], list[Optimizer]] | None:
        """Call AnomlibModule's configure optimizer."""
        return super().configure_optimizers()  # type: ignore[misc]

    def on_train_epoch_end(self) -> None:
        """Callback triggered when the training epoch ends."""
        return super().on_train_epoch_end()  # type: ignore[misc]

    def on_validation_start(self) -> None:
        """Callback triggered when the validation starts."""
        return super().on_validation_start()  # type: ignore[misc]

    def training_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call training step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)  # type: ignore[attr-defined]
        return super().training_step(inputs, batch_idx)  # type: ignore[misc]

    def validation_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call validation step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)  # type: ignore[attr-defined]
        return super().validation_step(inputs, batch_idx)  # type: ignore[misc]

    def test_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Call test step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)  # type: ignore[attr-defined]
        return super().test_step(inputs, batch_idx, **kwargs)  # type: ignore[misc]

    def predict_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Call test step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)  # type: ignore[attr-defined]
        return super().predict_step(inputs, batch_idx, **kwargs)  # type: ignore[misc]

    def forward(
        self,
        inputs: AnomalyModelInputs,
    ) -> AnomalyModelOutputs:
        """Wrap forward method of the Anomalib model."""
        outputs = self.validation_step(inputs)
        # TODO(Ashwin): update forward implementation to comply with other OTX models
        _PostProcessorCallback._post_process(outputs)  # noqa: SLF001
        _PostProcessorCallback._compute_scores_and_labels(self, outputs)  # noqa: SLF001
        _MinMaxNormalizationCallback._normalize_batch(outputs, self)  # noqa: SLF001

        return self._customize_outputs(outputs=outputs, inputs=inputs)  # type: ignore[attr-defined]
