# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientNetV2 model implementation."""

from __future__ import annotations

from copy import copy, deepcopy
from math import ceil
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from otx.algo.classification.backbones.timm import TimmBackbone, TimmModelType
from otx.algo.classification.classifier import HLabelClassifier, ImageClassifier, SemiSLClassifier
from otx.algo.classification.heads import (
    HierarchicalCBAMClsHead,
    LinearClsHead,
    MultiLabelLinearClsHead,
    SemiSLLinearClsHead,
)
from otx.algo.classification.losses.asymmetric_angular_loss_with_ignore import AsymmetricAngularLossWithIgnore
from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.algo.classification.utils import get_classification_layers
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.metrics.accuracy import HLabelClsMetricCallable, MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import (
    OTXHlabelClsModel,
    OTXMulticlassClsModel,
    OTXMultilabelClsModel,
)
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo, LabelInfoTypes
from otx.core.types.task import OTXTrainType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class TimmModelForMulticlassCls(OTXMulticlassClsModel):
    """TimmModel for multi-class classification task."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        backbone: TimmModelType,
        input_size: tuple[int, int] = (224, 224),  # input size of default classification data recipe
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
        train_type: Literal[OTXTrainType.SUPERVISED, OTXTrainType.SEMI_SUPERVISED] = OTXTrainType.SUPERVISED,
    ) -> None:
        self.backbone = backbone
        self.pretrained = pretrained

        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            train_type=train_type,
        )

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        model = self._build_model(num_classes=self.num_classes)
        model.init_weights()
        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = TimmBackbone(backbone=self.backbone, pretrained=self.pretrained)
        neck = GlobalAveragePooling(dim=2)
        if self.train_type == OTXTrainType.SEMI_SUPERVISED:
            return SemiSLClassifier(
                backbone=backbone,
                neck=neck,
                head=SemiSLLinearClsHead(
                    num_classes=num_classes,
                    in_channels=backbone.num_features,
                ),
                loss=nn.CrossEntropyLoss(reduction="none"),
            )

        return ImageClassifier(
            backbone=backbone,
            neck=neck,
            head=LinearClsHead(
                num_classes=num_classes,
                in_channels=backbone.num_features,
            ),
            loss=nn.CrossEntropyLoss(),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_v2_ckpt(state_dict, "multiclass", add_prefix)

    def forward_explain(self, inputs: MulticlassClsBatchDataEntity) -> MulticlassClsBatchPredEntity:
        """Model forward explain function."""
        outputs = self.model(images=inputs.stacked_images, mode="explain")

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs["preds"]),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["preds"],
            scores=outputs["scores"],
            saliency_map=outputs["saliency_map"],
            feature_vector=outputs["feature_vector"],
        )

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")


class TimmModelForMultilabelCls(OTXMultilabelClsModel):
    """TimmModel for multi-label classification task."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        backbone: TimmModelType,
        input_size: tuple[int, int] = (224, 224),  # input size of default classification data recipe
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.backbone = backbone
        self.pretrained = pretrained

        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        model = self._build_model(num_classes=self.num_classes)
        model.init_weights()
        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = TimmBackbone(backbone=self.backbone, pretrained=self.pretrained)
        return ImageClassifier(
            backbone=backbone,
            neck=GlobalAveragePooling(dim=2),
            head=MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=backbone.num_features,
                normalized=True,
            ),
            loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
            loss_scale=7.0,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_v2_ckpt(state_dict, "multilabel", add_prefix)

    def forward_explain(self, inputs: MultilabelClsBatchDataEntity) -> MultilabelClsBatchPredEntity:
        """Model forward explain function."""
        outputs = self.model(images=inputs.stacked_images, mode="explain")

        return MultilabelClsBatchPredEntity(
            batch_size=len(outputs["preds"]),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["preds"],
            scores=outputs["scores"],
            saliency_map=outputs["saliency_map"],
            feature_vector=outputs["feature_vector"],
        )

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")


class TimmModelForHLabelCls(OTXHlabelClsModel):
    """EfficientNetV2 Model for hierarchical label classification task."""

    label_info: HLabelInfo

    def __init__(
        self,
        label_info: HLabelInfo,
        backbone: TimmModelType,
        input_size: tuple[int, int] = (224, 224),  # input size of default classification data recipe
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.backbone = backbone
        self.pretrained = pretrained

        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_config = deepcopy(self.label_info.as_head_config_dict())
        sample_config["num_classes"] = 5
        sample_model_dict = self._build_model(head_config=sample_config).state_dict()
        sample_config["num_classes"] = 6
        incremental_model_dict = self._build_model(head_config=sample_config).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        model = self._build_model(head_config=self.label_info.as_head_config_dict())
        model.init_weights()
        return model

    def _build_model(self, head_config: dict) -> nn.Module:
        backbone = TimmBackbone(backbone=self.backbone, pretrained=self.pretrained)
        copied_head_config = copy(head_config)
        copied_head_config["step_size"] = (ceil(self.input_size[0] / 32), ceil(self.input_size[1] / 32))
        return HLabelClassifier(
            backbone=backbone,
            neck=nn.Identity(),
            head=HierarchicalCBAMClsHead(
                in_channels=backbone.num_features,
                **copied_head_config,
            ),
            multiclass_loss=nn.CrossEntropyLoss(),
            multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_v2_ckpt(state_dict, "hlabel", add_prefix)

    def forward_explain(self, inputs: HlabelClsBatchDataEntity) -> HlabelClsBatchPredEntity:
        """Model forward explain function."""
        outputs = self.model(images=inputs.stacked_images, mode="explain")

        return HlabelClsBatchPredEntity(
            batch_size=len(outputs["preds"]),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["preds"],
            scores=outputs["scores"],
            saliency_map=outputs["saliency_map"],
            feature_vector=outputs["feature_vector"],
        )

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
