"""Anomaly Lightning OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import onnx
import openvino
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
from otx.core.exporter.base import OTXModelExporter
from otx.core.types.export import OTXExportFormatType, TaskLevelExportParameters
from otx.core.types.label import NullLabelInfo
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from anomalib.metrics import AnomalibMetricCollection
    from anomalib.metrics.threshold import BaseThreshold
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks.callback import Callback
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torchmetrics import Metric
    from torchvision.transforms.v2 import Transform


AnomalyModelInputs: TypeAlias = (
    AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch
)
AnomalyModelOutputs: TypeAlias = (
    AnomalyClassificationBatchPrediction | AnomalySegmentationBatchPrediction | AnomalyDetectionBatchPrediction
)


class _AnomalyModelExporter(OTXModelExporter):
    def __init__(
        self,
        image_shape: tuple[int, int] = (256, 256),
        image_threshold: float = 0.5,
        pixel_threshold: float = 0.5,
        task: AnomalibTaskType = AnomalibTaskType.CLASSIFICATION,
        # the actual values for mean and scale should be in range 0-255
        mean_values: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale_values: tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalization_scale: float = 1.0,
    ) -> None:
        self.orig_height, self.orig_width = image_shape
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold
        self.task = task
        self.normalization_scale = normalization_scale

        super().__init__(
            task_level_export_parameters=TaskLevelExportParameters(
                model_type="anomaly",
                task_type="anomaly",
                label_info=NullLabelInfo(),
                optimization_config={},
            ),
            input_size=(1, 3, *image_shape),
            mean=mean_values,
            std=scale_values,
            swap_rgb=False,  # default value. Ideally, modelAPI should pass RGB inputs after the pre-processing step
        )

    @property
    def metadata(self) -> dict[tuple[str, str], str | float | int | tuple[int, int]]:  # type: ignore[override]
        return {
            ("model_info", "image_threshold"): self.image_threshold,
            ("model_info", "pixel_threshold"): self.pixel_threshold,
            ("model_info", "normalization_scale"): self.normalization_scale,
            ("model_info", "orig_height"): self.orig_height,
            ("model_info", "orig_width"): self.orig_width,
            ("model_info", "image_shape"): (self.orig_height, self.orig_width),
            ("model_info", "labels"): "Normal Anomaly",
            ("model_info", "model_type"): "AnomalyDetection",
            ("model_info", "task"): self.task.value,
        }

    def to_openvino(
        self,
        model: nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        save_path = str(output_dir / f"{base_model_name}.xml")
        exported_model = openvino.convert_model(
            input_model=model,
            example_input=torch.rand(self.input_size),
            input=(openvino.runtime.PartialShape(self.input_size)),
        )
        exported_model = self._postprocess_openvino_model(exported_model)
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXPrecisionType.FP16))
        return Path(save_path)

    def to_onnx(
        self,
        model: nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        save_path = str(output_dir / f"{base_model_name}.onnx")
        torch.onnx.export(
            model=model,
            args=(torch.rand(1, 3, self.orig_height, self.orig_width)).to(
                next(model.parameters()).device,
            ),
            f=save_path,
            opset_version=14,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            input_names=["input"],
            output_names=["output"],
        )
        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)
        onnx.save(onnx_model, save_path)
        return Path(save_path)


class OTXAnomaly:
    """Methods used to make OTX model compatible with the Anomalib model."""

    def __init__(self) -> None:
        self.optimizer: list[OptimizerCallable] | OptimizerCallable = None
        self.scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = None
        self._input_size: tuple[int, int] = (256, 256)
        self.mean_values: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.scale_values: tuple[float, float, float] = (1.0, 1.0, 1.0)
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

        attrs = ["_task_type", "_input_size", "mean_values", "scale_values", "image_threshold", "pixel_threshold"]

        checkpoint["anomaly"] = {key: getattr(self, key, None) for key in attrs}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on loading checkpoint."""
        super().on_load_checkpoint(checkpoint)  # type: ignore[misc]

        if anomaly_attrs := checkpoint.get("anomaly", None):
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

    def _extract_mean_scale_from_transforms(self, transforms: list[Transform]) -> None:
        """Extract mean and scale values from transforms."""
        for transform in transforms:
            name = transform.__class__.__name__
            if "Resize" in name:
                self.input_size = transform.size * 2  # transform.size has value [size], so *2 gives (size, size)
            elif "Normalize" in name:
                self.mean_values = transform.mean
                self.scale_values = transform.std

    @property
    def trainable_model(self) -> str | None:
        """Use this to return the name of the model that needs to be trained.

        This might not be the cleanest solution.

        Some models have multiple architectures and only one of them needs to be trained.
        However the optimizer is configured in the Anomalib's lightning model. This can be used
        to inform the OTX lightning model which model to train.
        """
        return None

    def setup(self, stage: str | None = None) -> None:
        """Setup the model."""
        super().setup(stage)  # type: ignore[misc]
        if stage == "fit" and hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "config"):
            if hasattr(self.trainer.datamodule.config, "test_subset"):
                self._extract_mean_scale_from_transforms(self.trainer.datamodule.config.test_subset.transforms)
            elif hasattr(self.trainer.datamodule.config, "val_subset"):
                self._extract_mean_scale_from_transforms(self.trainer.datamodule.config.val_subset.transforms)

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

    def on_test_batch_end(
        self,
        outputs: dict,
        batch: AnomalyModelInputs | dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called in the predict loop after the batch.

        Args:
            outputs: The outputs of predict_step(x)
            batch: The batched data as it is returned by the prediction DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """
        if not isinstance(batch, dict):
            batch = self._customize_inputs(batch)
        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)  # type: ignore[misc]

    def predict_step(
        self,
        inputs: AnomalyModelInputs | dict,
        batch_idx: int = 0,
        **kwargs,
    ) -> dict:
        """Return predictions from the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return super().predict_step(inputs, batch_idx, **kwargs)  # type: ignore[misc]

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

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.Optimizer]] | None:  # type: ignore[override]
        """Configure optimizers for Anomalib models.

        If the anomalib lightning model supports optimizers, return the optimizer.
        If ``self.trainable_model`` is None then the model does not support training.
        Else don't return optimizer even if it is configured in the OTX model.
        """
        # [TODO](ashwinvaidya17): Revisit this method
        if self.optimizer and self.trainable_model:
            optimizer = self.optimizer
            if isinstance(optimizer, list):
                if len(optimizer) > 1:
                    msg = "Only one optimizer should be passed"
                    raise ValueError(msg)
                optimizer = optimizer[0]
            params = getattr(self.model, self.trainable_model).parameters()
            return optimizer(params=params)
        return super().configure_optimizers()  # type: ignore[misc]

    def forward(
        self,
        inputs: AnomalyModelInputs,
    ) -> AnomalyModelOutputs:
        """Wrap forward method of the Anomalib model."""
        _inputs: dict = self._customize_inputs(inputs)
        outputs = self.model.model.forward(_inputs)
        return self._customize_outputs(outputs=outputs, inputs=inputs)

    def _customize_inputs(
        self,
        inputs: AnomalyModelInputs,
    ) -> dict[str, Any]:
        """Customize inputs for the model."""
        if isinstance(inputs, AnomalyClassificationDataBatch):
            return {"image": inputs.images, "label": torch.vstack(inputs.labels).squeeze()}
        if isinstance(inputs, AnomalySegmentationDataBatch):
            return {"image": inputs.images, "label": torch.vstack(inputs.labels).squeeze(), "mask": inputs.masks}
        if isinstance(inputs, AnomalyDetectionDataBatch):
            return {
                "image": inputs.images,
                "label": torch.vstack(inputs.labels).squeeze(),
                "mask": inputs.masks,
                "boxes": inputs.boxes,
            }
        msg = f"Unsupported input type {type(inputs)}"
        raise ValueError(msg)

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

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model

        Returns:
            Path: path to the exported model.
        """
        min_val = self.normalization_metrics.state_dict()["min"].cpu().numpy().tolist()
        max_val = self.normalization_metrics.state_dict()["max"].cpu().numpy().tolist()
        image_shape = (256, 256) if self.input_size is None else self.input_size
        exporter = _AnomalyModelExporter(
            image_shape=image_shape,
            image_threshold=self.image_threshold.value.cpu().numpy().tolist(),
            pixel_threshold=self.pixel_threshold.value.cpu().numpy().tolist(),
            task=self.task,
            mean_values=self.mean_values,
            scale_values=self.scale_values,
            normalization_scale=max_val - min_val,
        )
        return exporter.export(
            model=self.model,
            output_dir=output_dir,
            base_model_name=base_name,
            export_format=export_format,
            precision=precision,
        )
