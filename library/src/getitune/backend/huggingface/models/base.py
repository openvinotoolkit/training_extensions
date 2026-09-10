# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base model contract for the Hugging Face backend.

There is one wrapper per task rather than one per model. The ``transformers``
training contract is stable within a task, so adding a checkpoint is normally
just a recipe entry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
import transformers
from torch import nn
from transformers.utils import ModelOutput

from getitune.backend.huggingface.exporter.hf_exporter import HFModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.types.export import ExportFormat, TaskLevelExportParameters
from getitune.types.label import LabelInfo
from getitune.types.precision import Precision

if TYPE_CHECKING:
    from torchmetrics import Metric, MetricCollection
    from transformers.image_processing_utils import BaseImageProcessor

    from getitune.backend.lightning.exporter.base import ModelExporter
    from getitune.config.data import IntensityConfig
    from getitune.data.entity.sample import PredictionBatch, SampleBatch
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes
    from getitune.types.task import TaskType

__all__ = ["HFModel", "ModelOutput", "transformers"]


class HFModel(ABC, nn.Module):
    """Wraps a ``transformers`` computer-vision model for one Geti task.

    Subclasses override ``build_targets``, ``postprocess``, ``to_metric_inputs``,
    ``forward_for_tracing``, and ``build_default_metric``. Everything else,
    including ``forward``, is shared.

    Class attributes:
        task: The task this wrapper serves.
        hf_auto_class: ``Auto*`` class used to build the underlying model,
            e.g. ``AutoModelForObjectDetection``.
        export_model_type: ModelAPI ``model_info/model_type`` string written
            during export, e.g. ``"ssd"`` or ``"DETRInstSeg"``.
        label_keys: Target keyword names the underlying model expects. Passed
            to ``TrainingArguments(label_names=...)``.
        _onnx_output_names: Output tensor names baked into the ONNX export;
            ModelAPI's parsers resolve outputs by name.
    """

    task: ClassVar[TaskType]
    hf_auto_class: ClassVar[type]
    export_model_type: ClassVar[str] = "null"
    label_keys: ClassVar[tuple[str, ...]] = ("labels",)
    _onnx_output_names: ClassVar[list[str]] = []

    def __init__(
        self,
        checkpoint: str | transformers.PretrainedConfig,
        label_info: LabelInfoTypes,
        *,
        data_input_params: DataInputParams | dict[str, Any] | None = None,
        resize_mode: Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"] = "standard",
        pretrained: bool = True,
        extra_overrides: dict[str, Any] | None = None,
        onnx_dynamo: bool | None = None,
    ) -> None:
        """Build the underlying ``transformers`` model.

        Args:
            checkpoint: A Hub repo id, a local ``save_pretrained()``
                directory, or an already-built ``PretrainedConfig``. Passing a
                config builds the model from scratch with random weights and
                *pretrained* is ignored; this is the offline path used by
                tests and by recipes that define a model from scratch. Also
                doubles as the key into ``_default_preprocessing_params``.
            label_info: Label metadata, or anything ``_dispatch_label_info``
                accepts (an int, a list of names, or a ``LabelInfo``).
            data_input_params: Input size/mean/std used for export metadata
                (Geti's data pipeline does the actual resizing). A full or
                partial dict is merged with this checkpoint's entry in
                ``_default_preprocessing_params``; ``None`` uses that entry
                as-is. Matches ``LightningModel``'s ``data_input_params``.
            resize_mode: Resize mode used by the data pipeline and recorded in
                exported model metadata.
            pretrained: Load Hub/local weights if ``True``, otherwise build an
                untrained model from the resolved config.
            extra_overrides: Extra keyword arguments forwarded to
                ``from_pretrained`` / ``from_config``, e.g. ``problem_type`` or
                ``semantic_loss_ignore_index``.
            onnx_dynamo: override for ``torch.onnx.export(...,
                dynamo=...)``. ``None`` (default) leaves the exporter's own
                default (``True``) in place; set explicitly per recipe for
                checkpoints that need the other mode.
        """
        super().__init__()
        self.checkpoint = checkpoint if isinstance(checkpoint, str) else type(checkpoint).__name__
        self.pretrained = pretrained
        self.extra_overrides = dict(extra_overrides or {})
        self._onnx_dynamo = onnx_dynamo
        self._label_info = self._dispatch_label_info(label_info)
        self._data_input_params = self._configure_preprocessing_params(data_input_params)
        self._resize_mode = resize_mode
        self._intensity_config: IntensityConfig | None = None
        self._best_checkpoint: Path | None = None

        id2label = dict(enumerate(self._label_info.label_names))
        label2id = {name: idx for idx, name in id2label.items()}

        if isinstance(checkpoint, transformers.PretrainedConfig):
            for key, value in {"id2label": id2label, "label2id": label2id, **self.extra_overrides}.items():
                setattr(checkpoint, key, value)
            self.hf_model = self.hf_auto_class.from_config(checkpoint)
        elif pretrained:
            self.hf_model = self.hf_auto_class.from_pretrained(
                checkpoint,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
                **self.extra_overrides,
            )
        else:
            config = transformers.AutoConfig.from_pretrained(
                checkpoint,
                id2label=id2label,
                label2id=label2id,
                **self.extra_overrides,
            )
            self.hf_model = self.hf_auto_class.from_config(config)

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> LabelInfo:
        """Normalize *label_info* to a :class:`LabelInfo`.

        Accepts a dict, a plain int (number of classes), a list of label names, or a
        ``LabelInfo`` instance (passed through, including subclasses such as
        ``SegLabelInfo``).
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
            return label_info
        msg = f"Cannot build LabelInfo from {label_info!r}"
        raise TypeError(msg)

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        """Default preprocessing parameters, by checkpoint if task-specific.

        Task subclasses override this with a dict of known checkpoints fall back
        to the generic default below.
        """
        return DataInputParams(input_size=(640, 640), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def _configure_preprocessing_params(
        self,
        preprocessing_params: DataInputParams | dict[str, Any] | None,
    ) -> DataInputParams:
        """Resolve *preprocessing_params* against this checkpoint's registered defaults."""
        defaults = self._default_preprocessing_params
        if isinstance(defaults, dict):
            default = defaults.get(
                self.checkpoint,
                DataInputParams(input_size=(640, 640), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            )
        else:
            default = defaults

        if isinstance(preprocessing_params, DataInputParams):
            return preprocessing_params
        if isinstance(preprocessing_params, dict):
            return DataInputParams(
                input_size=preprocessing_params.get("input_size") or default.input_size,
                mean=preprocessing_params.get("mean") or default.mean,
                std=preprocessing_params.get("std") or default.std,
            )
        if preprocessing_params is None:
            return default
        msg = f"data_input_params should be DataInputParams, dict, or None, got {type(preprocessing_params)}."
        raise TypeError(msg)

    @abstractmethod
    def build_targets(self, batch: SampleBatch) -> dict[str, Any]:
        """Convert a Geti batch into ``transformers`` forward kwargs.

        Args:
            batch: Collated Geti batch. Every target field is a Python list,
                not a stacked tensor.

        Returns:
            Forward kwargs, always including ``pixel_values``.
        """
        raise NotImplementedError

    def forward(self, batch: SampleBatch) -> ModelOutput:
        """Run a training forward pass.

        Shared across tasks: only ``build_targets`` differs per task.
        """
        return self.hf_model(**self.build_targets(batch))

    def build_eval_inputs(self, batch: SampleBatch) -> dict[str, Any]:
        """Build the minimal forward kwargs an eval/predict pass needs."""
        return {"pixel_values": batch.images}

    @abstractmethod
    def postprocess(self, outputs: ModelOutput, batch: SampleBatch) -> PredictionBatch:
        """Turn raw model outputs into Geti predictions."""
        raise NotImplementedError

    @abstractmethod
    def to_metric_inputs(self, outputs: ModelOutput, batch: SampleBatch) -> dict[str, Any]:
        """Build the keyword arguments the task's Geti metric callable expects.

        Returns:
            A dict such as ``{"preds": ..., "target": ...}``, passed to the
            metric as ``metric.update(**result)``. The shapes inside differ
            per task (lists of per-image dicts for detection and instance
            segmentation, plain tensors for the rest) because that is what
            each underlying torchmetrics metric actually expects — this
            method exists precisely to hide that difference from the caller.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_for_tracing(self, images: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        """Return export-shaped outputs for ModelAPI.

        Args:
            images: Batched input shaped ``(B, C, H, W)``.

        Returns:
            Either a plain tensor (classification, semantic segmentation) or
            an output-name-to-tensor dict (detection, instance segmentation).
            Dict keys become the ONNX ``output_names`` and must be passed
            explicitly at export time, otherwise ONNX numbers them and
            ModelAPI's detection parser cannot match them.
        """
        raise NotImplementedError

    @abstractmethod
    def build_default_metric(self) -> Metric | MetricCollection:
        """Build this task's default Geti metric, ready for ``.update()``.

        Used by ``HFEngine.test()`` unless the caller supplies its own metric
        callable. A method rather than a bare class-level callable because
        semantic segmentation needs the model's resolved ``ignore_index``,
        which is only known once the model is constructed.
        """
        raise NotImplementedError

    def set_intensity_config(self, intensity_config: IntensityConfig | None) -> None:
        """Attach the DataModule intensity config so export metadata is correct."""
        self._intensity_config = intensity_config

    def ensure_predict_ready(self) -> None:
        """Load the image processor if needed and switch to eval mode.

        Subclasses that override this must call ``super()``.
        """
        self.eval()

    def load_checkpoint(self, checkpoint: PathLike) -> None:
        """Reload weights from a ``save_pretrained()`` directory.

        Replaces the wrapped model in place and records the checkpoint so
        ``best_checkpoint`` reflects it.
        """
        self.hf_model = self.hf_auto_class.from_pretrained(str(checkpoint))
        self._best_checkpoint = Path(checkpoint)

    def record_checkpoint(self, checkpoint: PathLike) -> None:
        """Record *checkpoint* as the location backing this model's current weights.

        Unlike :meth:`load_checkpoint`, this does not touch ``hf_model``. Used
        right after training, once the in-memory weights already match what
        was just written to *checkpoint* by ``Trainer.save_model()``, so
        reloading them from disk would be pure waste.
        """
        self._best_checkpoint = Path(checkpoint)

    @property
    def label_info(self) -> LabelInfo:
        """Label metadata backing ``id2label`` and ``label2id``."""
        return self._label_info

    @property
    def imgsz(self) -> int:
        """Square input size, for recipe convenience.

        ``data_input_params.input_size`` is the source of truth.
        """
        return self.data_input_params.input_size[0]

    @property
    def data_input_params(self) -> DataInputParams:
        """Preprocessing parameters used for export metadata."""
        return self._data_input_params

    @property
    def resize_mode(self) -> Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"]:
        """Resize mode used by the data pipeline."""
        return self._resize_mode

    @property
    def best_checkpoint(self) -> Path | None:
        """Most recently saved checkpoint directory, if any."""
        return self._best_checkpoint

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Export parameters shared at the task level.

        Task subclasses override this with ``.wrap(...)``, following the same
        pattern as ``LightningModel._export_parameters``.
        """
        return TaskLevelExportParameters(
            model_type="null",
            task_type="null",
            model_name=self.checkpoint,
            label_info=self.label_info,
            optimization_config={},
        )

    @cached_property
    def _image_processor(self) -> BaseImageProcessor:
        """Load the checkpoint's image processor for post-processing."""
        return transformers.AutoImageProcessor.from_pretrained(
            self.checkpoint,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            do_pad=False,
        )

    @property
    def _exporter(self) -> ModelExporter:
        """Build the ONNX/OpenVINO exporter from the task's output names.

        Task subclasses set ``_onnx_output_names`` and inherit this builder,
        keeping the export contract (input names, resize mode, swap RGB,
        opset) in one place. ``dynamo`` defaults to the exporter's own
        default (``False``) unless overridden via ``onnx_dynamo``.
        """
        if not hasattr(self, "_onnx_output_names") or not self._onnx_output_names:
            msg = "ONNX output names are not set."
            raise ValueError(msg)

        onnx_export_configuration: dict[str, Any] = {
            "input_names": ["images"],
            "output_names": self._onnx_output_names,
        }
        if self._onnx_dynamo is not None:
            onnx_export_configuration["dynamo"] = self._onnx_dynamo

        return HFModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode=self.resize_mode,
            swap_rgb=False,
            onnx_export_configuration=onnx_export_configuration,
        )

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: ExportFormat,
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export this model to OpenVINO IR or ONNX.

        Traces via ``forward_for_tracing`` rather than the training
        ``forward``: swaps ``self.forward`` for the tracing duration only,
        the same pattern ``LightningModel.export`` uses. Weights are moved
        to CPU/fp32 first — training may leave the model on XPU, and tracing
        a model whose parameters live on an accelerator not available at
        export time (or in bf16, which ONNX/OpenVINO conversion tools do not
        reliably support) is not a risk worth taking for a one-off export
        call.

        Args:
            output_dir: Directory to write the exported artifact into.
            base_name: File stem for the exported artifact.
            export_format: ``ExportFormat.OPENVINO`` or ``ExportFormat.ONNX``.
            precision: Precision of the exported weights.

        Returns:
            Path to the exported model file.
        """
        mode = self.training
        original_device = next(self.hf_model.parameters()).device
        original_dtype = next(self.hf_model.parameters()).dtype

        self.eval()
        self.hf_model = self.hf_model.to(device="cpu", dtype=torch.float32)
        orig_forward = self.forward
        try:
            self.forward = self.forward_for_tracing  # type: ignore[method-assign]
            return self._exporter.export(
                self,  # pyrefly: ignore[bad-argument-type]
                output_dir,
                base_name,
                export_format,
                precision,
            )
        finally:
            self.forward = orig_forward  # type: ignore[method-assign]
            self.hf_model = self.hf_model.to(device=original_device, dtype=original_dtype)
            self.train(mode)
