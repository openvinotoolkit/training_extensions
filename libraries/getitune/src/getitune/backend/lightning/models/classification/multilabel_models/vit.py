# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ViT model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones.vision_transformer import VisionTransformerBackbone
from getitune.backend.lightning.models.classification.classifier import ImageClassifier
from getitune.backend.lightning.models.classification.heads import (
    MultiLabelLinearClsHead,
)
from getitune.backend.lightning.models.classification.losses import AsymmetricAngularLossWithIgnore
from getitune.backend.lightning.models.classification.multiclass_models.vit import ForwardExplainMixInForViT
from getitune.backend.lightning.models.classification.multilabel_models.base import (
    LightningMultilabelClsModel,
)
from getitune.backend.lightning.models.classification.utils.pretrained_urls import VIT_PRETRAINED_URLS
from getitune.backend.lightning.models.classification.utils.pretrained_weights import VisionTransformerWeightsLoader
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.metrics.accuracy import MultiLabelClsMetricCallable
from getitune.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from getitune.metrics import MetricCallable
    from getitune.types import PathLike


class VisionTransformerMultilabelCls(
    VisionTransformerWeightsLoader, ForwardExplainMixInForViT, LightningMultilabelClsModel
):
    """ViT Model for multi-label classification task."""

    pretrained_urls = VIT_PRETRAINED_URLS

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal[
            "vit-tiny",
            "vit-small",
            "vit-base",
            "vit-large",
            "dinov2-small",
            "dinov2-base",
            "dinov2-large",
            "dinov2-giant",
        ] = "vit-tiny",
        freeze_backbone: bool = False,
        peft: Literal["lora", "dora"] | None = None,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
    ) -> None:
        self.peft = peft

        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
        )

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        num_classes = num_classes if num_classes is not None else self.num_classes
        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
        vit_backbone = VisionTransformerBackbone(
            model_name=self.model_name,
            img_size=self.data_input_params.input_size,
            peft=self.peft,
        )
        model = ImageClassifier(
            backbone=vit_backbone,
            neck=None,
            head=MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=vit_backbone.embed_dim,
            ),
            loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        )
        model.init_weights()
        return model
