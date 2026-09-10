# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics semantic segmentation model wrapper."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.trainers.semantic_segmentation import SemanticSegmentationTrainer
from getitune.backend.ultralytics.validators.semantic_segmentation import SemanticSegmentationValidator
from getitune.config.data import IntensityConfig
from getitune.types.export import ExportFormat, TaskLevelExportParameters
from getitune.types.label import LabelInfo, LabelInfoTypes, SegLabelInfo
from getitune.types.precision import Precision

from .base import UltralyticsModel

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from getitune.backend.ultralytics.exporter import UltralyticsModelExporter


@contextmanager
def _force_logits_semantic_export() -> Iterator[None]:
    """Temporarily disable the Ultralytics semantic head's baked class map.

    Ultralytics branches the ``SemanticSegment`` head on the export format: for
    ``onnx``/``mnn``/``openvino`` it bakes ``argmax`` into the graph and emits a
    single-channel ``[B, H, W]`` class map. ModelAPI's ``SegmentationModel``
    cannot consume that layout — it misclassifies the 3D output as an
    ``[H, W, num_classes]`` tensor with ``num_classes == 1`` and fails on
    ``np.argmax(soft_prediction, axis=2)``. getitune's export contract
    (``return_soft_prediction=True``) requires float logits so ModelAPI can
    compute the hard prediction itself, so the head's format attribute is
    temporarily cleared during the raw export to force the logits branch.

    Upstream may add a configuration knob for this bake; this workaround should
    be replaced once it exists.
    """
    try:
        from ultralytics.nn.modules.head import SemanticSegment
    except ImportError:
        yield
        return
    original_forward = SemanticSegment.forward

    def forward_with_logits(
        self: SemanticSegment,
        x: object,
    ) -> object:
        original_format = self.format
        self.format = None
        try:
            return original_forward(self, x)
        finally:
            self.format = original_format

    SemanticSegment.forward = forward_with_logits  # pyrefly: ignore[assignment-type]
    try:
        yield
    finally:
        SemanticSegment.forward = original_forward


class UltralyticsSemanticSegModel(UltralyticsModel):
    """YOLO semantic segmentation model.

    Supported variants: ``yolo26n-sem``, ``yolo26s-sem``, ``yolo26m-sem``,
    ``yolo26l-sem``, ``yolo26x-sem``.
    """

    task: ClassVar[str] = "semantic"
    trainer_cls: ClassVar[type] = SemanticSegmentationTrainer
    validator_cls: ClassVar[type] = SemanticSegmentationValidator

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "yolo26n-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem.pt",
        "yolo26s-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-sem.pt",
        "yolo26m-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-sem.pt",
        "yolo26l-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-sem.pt",
        "yolo26x-sem": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-sem.pt",
    }

    metric_keys: ClassVar[dict[str, str]] = {
        "metrics/mIoU": "val/mIoU",
        "metrics/pixel_acc": "val/pixel_accuracy",
        "train/ce_loss": "train/ce_loss",
        "train/dice_loss": "train/dice_loss",
        "train/aux_loss": "train/aux_loss",
        "lr/pg0": "lr",
    }

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> SegLabelInfo:
        """Normalize label_info to a ``SegLabelInfo`` instance.

        Semantic segmentation needs the ignore_index field for proper handling
        of void pixels during metric computation and loss computation.
        """
        if isinstance(label_info, SegLabelInfo):
            return label_info
        if isinstance(label_info, dict):
            return SegLabelInfo(**label_info)
        if isinstance(label_info, int):
            return SegLabelInfo.from_num_classes(num_classes=label_info)
        if isinstance(label_info, (list, tuple)) and all(isinstance(name, str) for name in label_info):
            names = list(label_info)
            return SegLabelInfo(
                label_names=names,
                label_groups=[names],
                label_ids=[str(i) for i in range(len(names))],
            )
        if isinstance(label_info, LabelInfo):
            return SegLabelInfo(
                label_names=label_info.label_names,
                label_groups=label_info.label_groups,
                label_ids=label_info.label_ids,
            )
        raise TypeError(label_info)

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Per-variant preprocessing defaults.

        All YOLO26-sem models use 512x512 input with identity mean/std (no
        additional normalization after intensity scaling to [0, 1]).
        """
        default = DataInputParams(
            input_size=(512, 512),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            intensity_config=IntensityConfig(mode="scale_to_unit", storage_dtype="uint8"),
        )
        return {
            "yolo26n-sem": default,
            "yolo26s-sem": default,
            "yolo26m-sem": default,
            "yolo26l-sem": default,
            "yolo26x-sem": default,
        }

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Semantic segmentation export parameters."""
        return TaskLevelExportParameters(
            model_type="Segmentation",
            model_name=self.model_name,
            task_type="segmentation",
            label_info=self.label_info,
            optimization_config={},
            confidence_threshold=0.0,
            return_soft_prediction=True,
            blur_strength=-1,
            nms_execute=False,
        )

    @property
    def _exporter(self) -> UltralyticsModelExporter:
        """Build and return the model exporter with standard (non-letterbox) resize."""
        from getitune.backend.ultralytics.exporter import UltralyticsModelExporter

        return UltralyticsModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
        )

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: ExportFormat,
        precision: Precision = Precision.FP32,
        export_args: dict[str, object] | None = None,
    ) -> Path:
        """Export the model with float logits, never a baked argmax class map.

        Wraps the base export with ``_force_logits_semantic_export`` because the
        Ultralytics ``SemanticSegment`` head otherwise bakes ``argmax`` into the
        ONNX/OpenVINO graph, producing an output layout that ModelAPI's
        ``SegmentationModel`` rejects (see that context manager for details).

        Args and return mirror :meth:`UltralyticsModel.export`.
        """
        with _force_logits_semantic_export():
            return super().export(output_dir, base_name, export_format, precision, export_args)
