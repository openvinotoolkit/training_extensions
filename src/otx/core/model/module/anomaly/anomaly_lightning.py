"""Anomaly Lightning OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import openvino
import torch
from anomalib import TaskType
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization.min_max_normalization import _MinMaxNormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from torch import nn

from otx.core.data.dataset.base import LabelInfo
from otx.core.data.entity.anomaly import (
    AnomalyClassificationBatchPrediction,
    AnomalyClassificationDataBatch,
    AnomalyDetectionBatchPrediction,
    AnomalyDetectionDataBatch,
    AnomalySegmentationBatchPrediction,
    AnomalySegmentationDataBatch,
)
from otx.core.exporter.base import OTXModelExporter
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from collections import OrderedDict

    from anomalib.metrics import AnomalibMetricCollection
    from anomalib.metrics.threshold import BaseThreshold
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks.callback import Callback
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from torchmetrics import Metric
    from torchvision.transforms.v2 import Transform


class _AnomalyModelExporter(OTXModelExporter):
    def __init__(
        self,
        image_shape: tuple[int, int] = (256, 256),
        image_threshold: float = 0.5,
        pixel_threshold: float = 0.5,
        task: TaskType = TaskType.CLASSIFICATION,
        # the actual values for mean and scale should be in range 0-255
        mean_values: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale_values: tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalization_scale: float = 1.0,
    ) -> None:
        self.orig_height, self.orig_width = image_shape
        metadata = {
            ("model_info", "image_threshold"): image_threshold,
            ("model_info", "pixel_threshold"): pixel_threshold,
            ("model_info", "normalization_scale"): normalization_scale,
            ("model_info", "orig_height"): image_shape[0],
            ("model_info", "orig_width"): image_shape[1],
            ("model_info", "image_shape"): image_shape,
            ("model_info", "labels"): "Normal Anomaly",
            ("model_info", "model_type"): "AnomalyDetection",
            ("model_info", "task"): task.value,
        }
        super().__init__(
            input_size=(1, 3, *image_shape),
            mean=mean_values,
            std=scale_values,
            swap_rgb=False,  # default value. Ideally, modelAPI should pass RGB inputs after the pre-processing step
            metadata=metadata,
        )

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
    """Mixin to make OTX model compatible with the Anomalib model."""

    def __init__(self) -> None:
        self.optimizer: list[OptimizerCallable] | OptimizerCallable = None
        self.scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = None
        self.input_size: list[int] = [256, 256]
        self.mean_values: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.scale_values: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.trainer: Trainer
        self.model: nn.Module
        self.image_threshold: BaseThreshold
        self.pixel_threshold: BaseThreshold

        self.normalization_metrics: Metric

        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

    @property
    def task(self) -> TaskType:
        """Return the task type of the model."""
        if self._task_type:
            return self._task_type
        msg = "``self._task_type`` is not assigned"
        raise AttributeError(msg)

    @task.setter
    def task(self, value: OTXTaskType) -> None:
        if value == OTXTaskType.ANOMALY_CLASSIFICATION:
            self._task_type = TaskType.CLASSIFICATION
        elif value == OTXTaskType.ANOMALY_DETECTION:
            self._task_type = TaskType.DETECTION
        elif value == OTXTaskType.ANOMALY_SEGMENTATION:
            self._task_type = TaskType.SEGMENTATION
        else:
            msg = f"Unexpected task type: {value}"
            raise ValueError(msg)

    @property
    def label_info(self) -> LabelInfo:
        """Get this model label information."""
        return self._label_info

    @label_info.setter
    def label_info(self, value: LabelInfo | list[str]) -> None:
        """Set this model label information.

        It changes the number of classes to 2 and sets the labels as Normal and Anomaly.
        This is because Datumaro returns multiple classes from the dataset. If self.label_info != 2,
        then It will call self._reset_prediction_layer() to reset the prediction layer. Which is not required.

        This overrides the OTXModel's label_info setter.
        """
        if isinstance(value, list):
            # value can be greater than 2 as datumaro returns all anomalous categories separately
            self._label_info = LabelInfo(label_names=["Normal", "Anomaly"], label_groups=[["Normal", "Anomaly"]])
        else:
            self._label_info = value

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
        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "config"):
            if hasattr(self.trainer.datamodule.config, "test_subset"):
                self._extract_mean_scale_from_transforms(self.trainer.datamodule.config.test_subset.transforms)
            elif hasattr(self.trainer.datamodule.config, "val_subset"):
                self._extract_mean_scale_from_transforms(self.trainer.datamodule.config.val_subset.transforms)

    def configure_callbacks(self) -> list[Callback]:
        """Get all necessary callbacks required for training and post-processing on Anomalib models."""
        image_metrics = ["AUROC", "F1Score"]
        pixel_metrics = image_metrics if self.task != TaskType.CLASSIFICATION else None
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

    def training_step(
        self,
        inputs: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch | dict,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call training step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return super().training_step(inputs, batch_idx)  # type: ignore[misc]

    def validation_step(
        self,
        inputs: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch | dict,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call validation step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return super().validation_step(inputs, batch_idx)  # type: ignore[misc]

    def test_step(
        self,
        inputs: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch | dict,
        batch_idx: int = 0,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Call test step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return super().test_step(inputs, batch_idx, **kwargs)  # type: ignore[misc]

    def on_test_batch_end(
        self,
        outputs: dict,
        batch: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch | dict,
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
        inputs: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch | dict,
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
        batch: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch,
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

    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary of model entity with meta information.

        Returns:
            A dictionary containing datamodule state.

        """
        state_dict = super().state_dict()  # type: ignore[misc]
        # This is defined in OTXModel
        state_dict["meta_info"] = self.meta_info  # type: ignore[attr-defined]
        return state_dict

    def load_state_dict(self, ckpt: OrderedDict[str, Any], *args, **kwargs) -> None:
        """Pass the checkpoint to the anomaly model."""
        ckpt = ckpt.get("state_dict", ckpt)
        ckpt.pop("meta_info", None)  # [TODO](ashwinvaidya17): Revisit this method when OTXModel is the lightning model
        return super().load_state_dict(ckpt, *args, **kwargs)  # type: ignore[misc]

    def forward(
        self,
        inputs: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch,
    ) -> AnomalyClassificationBatchPrediction | AnomalySegmentationBatchPrediction | AnomalyDetectionBatchPrediction:
        """Wrap forward method of the Anomalib model."""
        _inputs: dict = self._customize_inputs(inputs)
        outputs = self.model.model.forward(_inputs)
        return self._customize_outputs(outputs=outputs, inputs=inputs)

    def _customize_inputs(
        self,
        inputs: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch,
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
        inputs: AnomalyClassificationDataBatch | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch,
    ) -> AnomalyClassificationBatchPrediction | AnomalySegmentationBatchPrediction | AnomalyDetectionBatchPrediction:
        if self.task == TaskType.CLASSIFICATION:
            return AnomalyClassificationBatchPrediction(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                labels=outputs["label"],
                # Note: this is the anomalous score. It should be inverted to report Normal score
                scores=outputs["pred_scores"],
                anomaly_maps=outputs["anomaly_maps"],
            )
        if self.task == TaskType.SEGMENTATION:
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
        if self.task == TaskType.DETECTION:
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
        exporter = _AnomalyModelExporter(
            image_shape=(self.input_size[0], self.input_size[1]),
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
