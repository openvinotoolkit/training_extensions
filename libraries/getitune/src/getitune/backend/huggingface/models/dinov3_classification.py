# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 classification wrappers using public Transformers checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import transformers
from torch import nn
from transformers.modeling_outputs import ImageClassifierOutput

from getitune.backend.huggingface.models.base import HFModel
from getitune.data.entity.sample import PredictionBatch
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfoTypes
from getitune.types.task import TaskType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers.utils import ModelOutput

    from getitune.data.entity.sample import SampleBatch

__all__ = ["HFDinov3MulticlassClsModel", "HFDinov3MultilabelClsModel"]

_GETITUNE_MODEL_TYPE_KEY = "getitune_model_type"
_GETITUNE_MODEL_TYPE = "dinov3_image_classifier"


class _DinoV3ImageClassifier(nn.Module):
    """A DINOv3 backbone with a Geti classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        config: transformers.PretrainedConfig,
        multilabel: bool,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        num_labels = config.num_labels
        if num_labels is None:
            msg = "DINOv3 classification config must define num_labels."
            raise ValueError(msg)
        hidden_size = getattr(backbone, "num_features", None) or getattr(config, "hidden_size", None)
        hidden_size = hidden_size or getattr(config, "num_features", None)
        if hidden_size is None:
            msg = "DINOv3 configuration must define hidden_size or num_features."
            raise ValueError(msg)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.multilabel = multilabel
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> ImageClassifierOutput:
        outputs = self.backbone(pixel_values=pixel_values)
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            pooled = pooler_output
        else:
            # Fall back through the shapes every backbone family can emit:
            # token sequences [B, tokens, C] take the CLS token, convolutional
            # backbones (e.g. timm ConvNeXt wrappers) give spatial [B, C, H, W]
            # tensors that need average pooling, and some wrappers hand back an
            # already-pooled [B, C] tensor directly.
            hidden_state = outputs.last_hidden_state  # pyrefly: ignore[missing-attribute]
            if hidden_state.ndim == 3:
                pooled = hidden_state[:, 0]
            elif hidden_state.ndim == 4:
                pooled = hidden_state.mean(dim=(2, 3))
            elif hidden_state.ndim == 2:
                pooled = hidden_state
            else:
                msg = (
                    f"Unsupported DINOv3 backbone pooling output shape {tuple(hidden_state.shape)} "
                    f"({hidden_state.ndim} dims); cannot derive a per-image feature vector."
                )
                raise ValueError(msg)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = (
                nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
                if self.multilabel
                else nn.functional.cross_entropy(logits, labels.long())
            )
        return ImageClassifierOutput(loss=cast("torch.FloatTensor | None", loss), logits=logits)

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        setattr(self.config, _GETITUNE_MODEL_TYPE_KEY, _GETITUNE_MODEL_TYPE)
        self.config.multilabel = self.multilabel
        self.config.freeze_backbone = self.freeze_backbone
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), path / "pytorch_model.bin")


class _DinoV3Factory:
    """Factory with the API expected by :class:`HFModel`."""

    @classmethod
    def from_config(cls, config: transformers.PretrainedConfig) -> nn.Module:
        backbone = transformers.AutoModel.from_config(config)
        return _DinoV3ImageClassifier(
            backbone,
            config,
            bool(getattr(config, "multilabel", False)),
            bool(getattr(config, "freeze_backbone", False)),
        )

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs: Any) -> nn.Module:  # noqa: ANN401
        freeze_backbone = bool(kwargs.pop("freeze_backbone", False))
        id2label = kwargs.pop("id2label", {})
        label2id = kwargs.pop("label2id", {label: index for index, label in id2label.items()})
        path = Path(checkpoint)
        config = transformers.AutoConfig.from_pretrained(checkpoint)
        if path.is_dir() and getattr(config, _GETITUNE_MODEL_TYPE_KEY, None) == _GETITUNE_MODEL_TYPE:
            state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
            model = cls.from_config(config)
            # The classifier head must be swapped BEFORE loading the saved state:
            # a label-count change reshapes ``classifier`` so the stored head has
            # a different shape and a strict load would raise a size mismatch.
            head_replaced = bool(id2label) and len(id2label) != config.num_labels
            if head_replaced:
                classifier = cast("nn.Linear", model.classifier)
                model.classifier = nn.Linear(classifier.in_features, len(id2label))
                msg = (
                    f"DINOv3 checkpoint '{checkpoint}' has {config.num_labels} labels "
                    f"but the project defines {len(id2label)}; the classifier head was "
                    f"re-initialized and trained from scratch."
                )
                logger.warning(msg)
                head_keys = {key for key in state_dict if key.startswith("classifier.")}
                for key in head_keys:
                    del state_dict[key]
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict)
            if id2label:
                config.id2label = id2label
                config.label2id = label2id
                config.num_labels = len(id2label)
            return model

        config.id2label = id2label
        config.label2id = label2id
        config.num_labels = len(id2label)
        config.multilabel = bool(kwargs.pop("multilabel", False))
        backbone = transformers.AutoModel.from_pretrained(checkpoint, **kwargs)
        return _DinoV3ImageClassifier(backbone, config, config.multilabel, freeze_backbone)


class _HFDinov3ClassificationModel(HFModel):
    """Shared DINOv3 classification implementation."""

    hf_auto_class: ClassVar[type] = _DinoV3Factory
    export_model_type: ClassVar[str] = "Classification"
    label_keys: ClassVar[tuple[str, ...]] = ("labels",)
    _onnx_output_names: ClassVar[list[str]] = ["logits"]
    multilabel: ClassVar[bool] = False
    freeze_backbone: ClassVar[bool] = False

    def __init__(
        self,
        checkpoint: str | transformers.PretrainedConfig,
        label_info: LabelInfoTypes,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        overrides = dict(kwargs.pop("extra_overrides", None) or {})
        overrides["multilabel"] = self.multilabel
        overrides.setdefault("freeze_backbone", self.freeze_backbone)
        super().__init__(checkpoint, label_info, extra_overrides=overrides, **kwargs)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        return super()._export_parameters.wrap(
            model_type=self.export_model_type,
            task_type="classification",
            multilabel=self.multilabel,
            output_raw_scores=True,
        )

    def build_targets(self, batch: SampleBatch) -> dict[str, Any]:
        if batch.labels is None:
            msg = "Classification batches need labels."
            raise ValueError(msg)
        labels = (
            torch.vstack(list(batch.labels)).float()
            if self.multilabel
            else torch.stack([x.long() for x in batch.labels])
        )
        return {"pixel_values": batch.images, "labels": labels}

    def postprocess(self, outputs: ModelOutput, batch: SampleBatch) -> PredictionBatch:
        logits = cast("torch.Tensor", outputs["logits"])
        scores = logits.sigmoid() if self.multilabel else logits.softmax(dim=-1)
        labels = (scores >= 0.5).long() if self.multilabel else scores.argmax(dim=-1)
        return PredictionBatch(images=batch.images, imgs_info=batch.imgs_info, labels=list(labels), scores=list(scores))

    def to_metric_inputs(self, outputs: ModelOutput, batch: SampleBatch) -> dict[str, Any]:
        if batch.labels is None:
            msg = "Classification batches need labels to compute metrics."
            raise ValueError(msg)
        logits = cast("torch.Tensor", outputs["logits"])
        return {
            "preds": logits.sigmoid() if self.multilabel else logits.argmax(dim=-1),
            "target": (
                torch.vstack(list(batch.labels)) if self.multilabel else torch.stack([x.long() for x in batch.labels])
            ),
        }

    def build_default_metric(self) -> Any:  # noqa: ANN401
        return (
            MultiLabelClsMetricCallable(self.label_info)
            if self.multilabel
            else MultiClassClsMetricCallable(self.label_info)
        )

    def forward_for_tracing(self, images: torch.Tensor) -> torch.Tensor:
        return self.hf_model(pixel_values=images).logits


class HFDinov3MulticlassClsModel(_HFDinov3ClassificationModel):
    """DINOv3 backbone for multi-class classification."""

    task: ClassVar[TaskType] = TaskType.MULTI_CLASS_CLS


class HFDinov3MultilabelClsModel(_HFDinov3ClassificationModel):
    """DINOv3 backbone for multi-label classification."""

    task: ClassVar[TaskType] = TaskType.MULTI_LABEL_CLS
    multilabel: ClassVar[bool] = True
