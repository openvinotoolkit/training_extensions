# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientNet-B0 model implementation."""


from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from torch import nn

from otx.algo.classification.backbones.efficientnet import EFFICIENTNET_VERSION, OTXEfficientNet
from otx.algo.classification.classifier.base_classifier import ImageClassifier
from otx.algo.classification.classifier.semi_sl_classifier import SemiSLClassifier
from otx.algo.classification.heads import (
    HierarchicalLinearClsHead,
    LinearClsHead,
    MultiLabelLinearClsHead,
    OTXSemiSLLinearClsHead,
)
from otx.algo.classification.losses.asymmetric_angular_loss_with_ignore import AsymmetricAngularLossWithIgnore
from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.algo.classification.utils import get_classification_layers
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.metrics.accuracy import DefaultClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import OTXClassificationModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo, LabelInfoTypes
from otx.core.types.task import OTXTaskType, OTXTrainType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallablePerTask


class EfficientNetForClassification(OTXClassificationModel):
    """EfficientNet Model for classification task."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        version: EFFICIENTNET_VERSION = "b0",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallablePerTask = DefaultClsMetricCallable,
        torch_compile: bool = False,
        task: Literal[
            OTXTaskType.MULTI_CLASS_CLS,
            OTXTaskType.MULTI_LABEL_CLS,
            OTXTaskType.H_LABEL_CLS,
        ] = OTXTaskType.MULTI_CLASS_CLS,
        train_type: Literal[OTXTrainType.SUPERVISED, OTXTrainType.SEMI_SUPERVISED] = OTXTrainType.SUPERVISED,
    ) -> None:
        self.version = version

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            task=task,
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
        backbone = OTXEfficientNet(version=self.version, pretrained=True)
        neck = GlobalAveragePooling(dim=2)
        classifier = ImageClassifier if self.train_type == OTXTrainType.SUPERVISED else SemiSLClassifier
        head = self._build_head(num_classes)

        return classifier(
            backbone=backbone,
            neck=neck,
            head=head,
        )

    def _build_head(self, num_classes: int) -> nn.Module:
        if self.task == OTXTaskType.MULTI_CLASS_CLS:
            loss = nn.CrossEntropyLoss(reduction="none")
            if self.train_type == OTXTrainType.SEMI_SUPERVISED:
                return OTXSemiSLLinearClsHead(
                    num_classes=num_classes,
                    in_channels=1280,
                    loss=loss,
                )
            return LinearClsHead(
                num_classes=num_classes,
                in_channels=1280,
                topk=(1, 5) if num_classes >= 5 else (1,),
                loss=loss,
            )
        if self.task == OTXTaskType.MULTI_LABEL_CLS:
            if self.train_type == OTXTrainType.SEMI_SUPERVISED:
                msg = "Semi-supervised learning is not supported for multi-label classification."
                raise NotImplementedError(msg)
            return MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=1280,
                scale=7.0,
                normalized=True,
                loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
            )
        if self.task == OTXTaskType.H_LABEL_CLS:
            if self.train_type == OTXTrainType.SEMI_SUPERVISED:
                msg = "Semi-supervised learning is not supported for h-label classification."
                raise NotImplementedError(msg)
            if not isinstance(self.label_info, HLabelInfo):
                msg = "LabelInfo should be HLabelInfo for H-label classification."
                raise ValueError(msg)
            head_config = deepcopy(self.label_info.as_head_config_dict())
            head_config["num_classes"] = num_classes
            return HierarchicalLinearClsHead(
                in_channels=1280,
                multiclass_loss=nn.CrossEntropyLoss(),
                multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
                **head_config,
            )
        msg = f"Unsupported task: {self.task}"
        raise NotImplementedError(msg)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        label_type = {
            OTXTaskType.MULTI_CLASS_CLS: "multiclass",
            OTXTaskType.MULTI_LABEL_CLS: "multilabel",
            OTXTaskType.H_LABEL_CLS: "hlabel",
        }[self.task]
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, label_type, add_prefix)
