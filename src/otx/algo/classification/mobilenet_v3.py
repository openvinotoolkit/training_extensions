# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MobileNetV3 model implementation."""


from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from torch import nn

from otx.algo.classification.backbones import OTXMobileNetV3
from otx.algo.classification.classifier import ImageClassifier, SemiSLClassifier
from otx.algo.classification.heads import (
    HierarchicalNonLinearClsHead,
    LinearClsHead,
    MultiLabelNonLinearClsHead,
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


class MobileNetV3ForClassification(OTXClassificationModel):
    """MobileNetV3ForClassification is a class that represents a MobileNetV3 model for classification.

    Args:
        label_info (LabelInfoTypes): The label information of OTX datamodule.
        mode (Literal["large", "small"]): The mode of the MobileNetV3 model, either "large" or "small".
        num_classes (int): The number of classes for classification.
        loss_callable (Callable[[], nn.Module], optional): The loss function callable. Defaults to nn.CrossEntropyLoss.
        optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallablePerTask, optional): The metric callable. Defaults to DefaultClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        task (Literal[OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS], optional):
            The task type. Defaults to OTXTaskType.MULTI_CLASS_CLS.
        train_type (Literal[OTXTrainType.SUPERVISED, OTXTrainType.SEMI_SUPERVISED], optional): The training type.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        mode: Literal["large", "small"] = "large",
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
        self.mode = mode

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
        backbone = OTXMobileNetV3(mode=self.mode)
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
                    in_channels=960,
                    loss=loss,
                )
            return LinearClsHead(
                num_classes=num_classes,
                in_channels=960,
                topk=(1, 5) if num_classes >= 5 else (1,),
                loss=loss,
            )
        if self.task == OTXTaskType.MULTI_LABEL_CLS:
            if self.train_type == OTXTrainType.SEMI_SUPERVISED:
                msg = "Semi-supervised learning is not supported for multi-label classification."
                raise NotImplementedError(msg)
            return MultiLabelNonLinearClsHead(
                num_classes=num_classes,
                in_channels=960,
                hid_channels=1280,
                normalized=True,
                scale=7.0,
                activation_callable=nn.PReLU(),
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
            return HierarchicalNonLinearClsHead(
                in_channels=960,
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
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, label_type, add_prefix)
