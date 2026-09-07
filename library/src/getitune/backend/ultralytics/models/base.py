# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for Ultralytics models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ultralytics import YOLO

from getitune.backend.lightning.models.base import DataInputParams
from getitune.config.data import IntensityConfig
from getitune.types.export import ExportFormat, TaskLevelExportParameters
from getitune.types.label import LabelInfo, LabelInfoTypes
from getitune.types.precision import Precision

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from getitune.backend.ultralytics.exporter import UltralyticsModelExporter
    from getitune.types import PathLike


class UltralyticsModel:
    """Wrapper around ``ultralytics.YOLO`` for :class:`UltralyticsEngine`.

    Args:
        model_name: Model variant identifier (e.g. ``"yolo26n"``).
        label_info: Label metadata — accepts the same types as
            ``LightningModel``: ``list[str]``, ``int``, ``LabelInfo``,
            or ``dict``.
        data_input_params: Optional preprocessing parameters. When provided,
            overrides the per-model defaults. Accepts a ``DataInputParams``
            instance or a plain ``dict`` (same as Lightning).
        pretrained: Whether to load pretrained weights.
        imgsz: Image size for training / inference.
        extra_overrides: Extra Ultralytics config forwarded to train/val/export.
        export_nms: Whether to embed NMS in the exported graph.
    """

    task: ClassVar[str] = ""
    metric_keys: ClassVar[dict[str, str]] = {}
    trainer_cls: ClassVar[type | None] = None
    validator_cls: ClassVar[type | None] = None

    _pretrained_weights: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        model_name: str,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        *,
        pretrained: bool = True,
        imgsz: int | None = None,
        extra_overrides: dict[str, Any] | None = None,
        export_nms: bool = False,
    ) -> None:
        self.model_name = model_name
        self.label_info = self._dispatch_label_info(label_info)
        self.pretrained = pretrained
        self.extra_overrides = extra_overrides or {}
        self.export_nms = export_nms

        if not self.model_name:
            msg = "model_name must be provided."
            raise ValueError(msg)

        if self.model_name.endswith(".pt") and not self.pretrained:
            msg = f"pretrained=False requires a model config (.yaml), not a checkpoint name: {self.model_name}"
            raise ValueError(msg)

        # Resolve image size: explicit arg > default from preprocessing params.
        if imgsz is not None:
            self.imgsz = imgsz
        else:
            default_params = self._default_preprocessing_params
            if isinstance(default_params, dict):
                params = default_params.get(self.model_name)
                self.imgsz = params.input_size[0] if params else 640
            else:
                self.imgsz = default_params.input_size[0]

        self._yolo: YOLO | None = None
        self._intensity_config: IntensityConfig | None = None
        self._export_args: dict[str, Any] = {}
        self._explicit_data_input_params = self._configure_preprocessing_params(data_input_params)

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> LabelInfo:
        """Normalize label_info to a ``LabelInfo`` instance.

        Accepts the same types as ``LightningModel._dispatch_label_info``.
        """
        if isinstance(label_info, dict):
            if "label_ids" not in label_info:
                label_info["label_ids"] = label_info["label_names"]
            return LabelInfo(**label_info)
        if isinstance(label_info, int):
            return LabelInfo.from_num_classes(num_classes=label_info)
        if isinstance(label_info, (list, tuple)) and all(isinstance(name, str) for name in label_info):
            return LabelInfo(
                label_names=list(label_info),
                label_groups=[list(label_info)],
                label_ids=[str(i) for i in range(len(label_info))],
            )
        if isinstance(label_info, LabelInfo):
            if not hasattr(label_info, "label_ids"):
                label_info.label_ids = label_info.label_names
            return label_info
        raise TypeError(label_info)

    def _configure_preprocessing_params(
        self,
        preprocessing_params: DataInputParams | dict | None = None,
    ) -> DataInputParams | None:
        """Normalize an optional preprocessing-params argument."""
        if preprocessing_params is None:
            return None
        if isinstance(preprocessing_params, dict):
            intensity_cfg = preprocessing_params.get("intensity_config")
            if isinstance(intensity_cfg, dict):
                intensity_cfg = IntensityConfig(**intensity_cfg)
            default = self._resolve_default_params()
            return DataInputParams(
                input_size=preprocessing_params.get("input_size") or default.input_size,
                mean=preprocessing_params.get("mean") or default.mean,
                std=preprocessing_params.get("std") or default.std,
                intensity_config=intensity_cfg if intensity_cfg is not None else default.intensity_config,
            )
        if isinstance(preprocessing_params, DataInputParams):
            return preprocessing_params
        return None

    @property
    def yolo(self) -> YOLO:
        """Lazily build and return the underlying ``ultralytics.YOLO`` instance."""
        if self._yolo is None:
            self._yolo = self._build_yolo()
        return self._yolo

    def _build_yolo(self) -> YOLO:
        """Create the ``ultralytics.YOLO`` model and optionally load pretrained weights."""
        config = self.model_name if self.model_name.endswith(".yaml") else f"{self.model_name}.yaml"
        logger.info(f"Building Ultralytics model: {config} (task={self.task})")
        yolo = YOLO(config, task=self.task or None)

        # Normalize model_name for pretrained weight lookup: strip .yaml/.yml
        # suffixes so "yolo26n.yaml" matches the "yolo26n" key in
        # _pretrained_weights.
        base_name = self.model_name.removesuffix(".yaml").removesuffix(".yml")
        if self.pretrained and base_name in self._pretrained_weights:
            weights_url = self._pretrained_weights[base_name]
            logger.info(f"Loading pretrained weights: {weights_url}")
            yolo.load(weights_url)
        elif self.pretrained and self._pretrained_weights:
            msg = (
                f"pretrained=True but no pretrained weights found for '{base_name}'. "
                f"Available: {list(self._pretrained_weights.keys())}"
            )
            raise ValueError(msg)

        return yolo

    def ensure_predict_ready(self) -> None:
        """Backfill state required by Ultralytics predictors before ``predict()``/``test()``.

        Default is a no-op. Subclasses override this when the underlying
        Ultralytics predictor/validator requires extra state that getitune's
        DataModule-driven training/inference path does not otherwise set up.
        """
        return

    def load_checkpoint(self, weights_path: PathLike) -> None:
        """Load weights from a local checkpoint file.

        Creates a fresh ``YOLO`` instance from the checkpoint so that any
        in-place layer fusion (performed during training/validation) is
        discarded.  This is required for reliable ``export()`` after
        ``train()``, because Ultralytics fuses Conv+BN layers in-place
        and the resulting model cannot be re-exported without reconstruction.

        Args:
            weights_path: Path to a checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(weights_path)
        if not path.exists():
            msg = f"Checkpoint file not found: {path}"
            raise FileNotFoundError(msg)
        logger.info(f"Loading checkpoint from: {path}")
        self._yolo = YOLO(str(path), task=self.task or None)

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: ExportFormat,
        precision: Precision = Precision.FP32,
        export_args: dict[str, Any] | None = None,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir: Directory for saving the exported model.
            base_name: Base name for the exported model file.
            export_format: Format of the output model.
            precision: Precision of the output model.
            export_args: Optional export arguments (confidence_threshold,
                iou_threshold) from the recipe to embed in model metadata.

        Returns:
            Path to the exported model file.
        """
        if export_args:
            self._export_args = export_args
        exporter = self._exporter
        if export_format == ExportFormat.OPENVINO:
            return exporter.to_openvino(self.yolo, output_dir, base_name, precision)
        if export_format == ExportFormat.ONNX:
            return exporter.to_onnx(self.yolo, output_dir, base_name, precision)
        msg = f"Unsupported export format: {export_format}"
        raise ValueError(msg)

    @property
    def _exporter(self) -> UltralyticsModelExporter:
        """Build and return the model exporter."""
        from getitune.backend.ultralytics.exporter import UltralyticsModelExporter

        return UltralyticsModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="fit_to_window_letterbox",
            pad_value=114,
            swap_rgb=False,
            export_nms=self.export_nms,
        )

    @property
    def data_input_params(self) -> DataInputParams:
        """Resolved data input parameters for this model instance."""
        if self._explicit_data_input_params is not None:
            return self._explicit_data_input_params

        params = self._resolve_default_params()

        # Resolve intensity config: engine-propagated > model default.
        intensity_config = self._intensity_config if self._intensity_config is not None else params.intensity_config

        if params.input_size != (self.imgsz, self.imgsz) or intensity_config is not params.intensity_config:
            return DataInputParams(
                input_size=(self.imgsz, self.imgsz),
                mean=params.mean,
                std=params.std,
                intensity_config=intensity_config,
            )
        return params

    def set_intensity_config(self, intensity_config: IntensityConfig) -> None:
        """Set runtime intensity config propagated from a DataModule."""
        self._intensity_config = intensity_config

    def _resolve_default_params(self) -> DataInputParams:
        """Return the default ``DataInputParams`` for the current ``model_name``."""
        default = self._default_preprocessing_params
        if isinstance(default, dict):
            params = default.get(self.model_name)
            if params is None:
                params = next(iter(default.values()))
            return params
        return default

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        """Per-model preprocessing parameters.

        Subclasses must override to provide model-specific defaults.
        """
        return DataInputParams(
            input_size=(self.imgsz, self.imgsz),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            intensity_config=IntensityConfig(mode="scale_to_unit", storage_dtype="uint8"),
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Task-level export parameters for metadata embedding.

        Subclasses override to set model_type, task_type, and thresholds.
        """
        iou = self._export_args.get("iou_threshold")
        if iou is None:
            iou = self.extra_overrides.get("iou", 0.5)
        conf = self._export_args.get("confidence_threshold")
        return TaskLevelExportParameters(
            model_type="YOLO11",
            model_name=self.model_name,
            task_type="detection",
            label_info=self.label_info,
            optimization_config={},
            confidence_threshold=float(conf) if conf is not None else None,
            iou_threshold=float(iou),
            nms_execute=not self.export_nms,
        )

    def __repr__(self) -> str:
        num_classes = self.label_info.num_classes
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"task={self.task!r}, "
            f"num_classes={num_classes}, "
            f"pretrained={self.pretrained})"
        )
