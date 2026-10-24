# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for keypoint detection model entity used in getitune."""

# type: ignore[override]

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch

from getitune.backend.lightning.models.base import (
    DataInputParams,
    DefaultOptimizerCallable,
    DefaultSchedulerCallable,
    LightningModel,
)
from getitune.backend.lightning.models.common.pretrained_weights import PretrainedWeightsMixin
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.data.entity.base import BatchLoss, ImageInfo
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.metrics import MetricCallable, MetricInput
from getitune.metrics.pck import PCKMeasureCallable
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfoTypes
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.types import PathLike


class LightningKeypointDetectionModel(PretrainedWeightsMixin, LightningModel):
    """Base class for the keypoint detection models used in getitune.

    Args:
        label_info (LabelInfoTypes | int | Sequence): Information about the labels used in the model.
            If `int` is given, label info will be constructed from number of classes,
            if `Sequence` is given, label info will be constructed from the sequence of label names.
        data_input_params (DataInputParams | dict | None, optional): Parameters for image data
            preprocessing. If None is given, default parameters for the specific model will be used.
        model_name (str, optional): Name of the model. Defaults to "keypoint_detection_model".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to PCKMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes | int | Sequence,
        data_input_params: DataInputParams | dict | None = None,
        model_name: str = "keypoint_detection_model",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = PCKMeasureCallable,
        torch_compile: bool = False,
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
    ) -> None:
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
        )

    def _customize_inputs(self, entity: SampleBatch) -> dict[str, Any]:
        """Convert TorchDataBatch into Topdown model's input."""
        inputs: dict[str, Any] = {}

        inputs["inputs"] = entity.images
        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"
        return inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: SampleBatch,
    ) -> PredictionBatch | BatchLoss:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = BatchLoss()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        keypoints = []
        scores = []
        # default visibility threshold
        visibility_threshold = 0.5
        if inputs.imgs_info is None:
            msg = "The input image information is not provided."
            raise ValueError(msg)
        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
        for i, output in enumerate(outputs):
            if not isinstance(output, tuple):
                raise TypeError(output)
            if inputs.imgs_info[i] is None:
                msg = f"The image information for the image {i} is not provided."
                raise ValueError(msg)
            # scale to the original image size
            orig_h, orig_w = inputs.imgs_info[i].ori_shape  # type: ignore[union-attr]
            kp_scale_h, kp_scale_w = (
                orig_h / self.data_input_params.input_size[0],
                orig_w / self.data_input_params.input_size[1],
            )
            inverted_scale = max(kp_scale_h, kp_scale_w)
            kp_scale_h = kp_scale_w = inverted_scale
            # decode kps
            kps = torch.as_tensor(output[0], dtype=torch.float32, device=self.device) * torch.tensor(
                [kp_scale_w, kp_scale_h],
                device=self.device,
            )
            score = torch.as_tensor(output[1], dtype=torch.float32, device=self.device)
            visible_keypoints = torch.cat([kps, score.unsqueeze(1) > visibility_threshold], dim=1)
            keypoints.append(visible_keypoints)
            scores.append(score)

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            keypoints=keypoints,
            scores=scores,
            bboxes=[],
            labels=[],
        )

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()
        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
        self._metric.input_size = tuple(self.data_input_params.input_size)

    def _convert_pred_entity_to_compute_metric(  # type: ignore[override]
        self,
        preds: PredictionBatch,
        inputs: SampleBatch,
    ) -> MetricInput:
        if inputs.keypoints is None:
            msg = "The input ground truth keypoints are not provided."
            raise ValueError(msg)

        if preds.keypoints is None or preds.scores is None:
            msg = "The predicted keypoints or scores are not provided."
            raise ValueError(msg)

        if len(preds.keypoints) != len(inputs.keypoints):
            msg = "The number of predicted keypoints and ground truth keypoints does not match."
            raise ValueError(msg)

        return {
            "preds": [
                {
                    "keypoints": kpt[:, :2],
                    "scores": score,
                }
                for kpt, score in zip(preds.keypoints, preds.scores)
            ],
            "target": [
                {
                    "keypoints": kpt[:, :2],
                    "keypoints_visible": kpt[:, 2],
                }
                for kpt in inputs.keypoints
            ],
        }

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model.forward(inputs=image, mode="tensor")

    def get_dummy_input(self, batch_size: int = 1) -> SampleBatch:  # type: ignore[override]
        """Generates a dummy input, suitable for launching forward() on it.

        Args:
            batch_size (int, optional): number of elements in a dummy input sequence. Defaults to 1.

        Returns:
            TorchDataBatch: An entity containing randomly generated inference data.
        """
        images = torch.rand(self.data_input_params.as_ncwh(batch_size))
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape[:2],
                    ori_shape=img.shape[:2],
                ),
            )

        return SampleBatch(
            images=images,
            labels=[],
            bboxes=[],
            keypoints=[],
            imgs_info=infos,  # type: ignore[arg-type]
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="keypoint_detection",
            task_type="keypoint_detection",
            confidence_threshold=self.hparams.get("best_confidence_threshold", None),
        )

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        return DataInputParams(input_size=(512, 512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    @property
    def task(self) -> TaskType:
        """Return task type."""
        return TaskType.KEYPOINT_DETECTION
