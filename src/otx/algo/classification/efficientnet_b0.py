# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""EfficientNetB0 model implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning.pytorch.cli import ReduceLROnPlateau

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.metrics.accuracy import HLabelClsMetricCallble, MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from otx.core.model.classification import (
    MMPretrainHlabelClsModel,
    MMPretrainMulticlassClsModel,
    MMPretrainMultilabelClsModel,
)
from otx.core.model.utils.mmpretrain import ExplainableMixInMMPretrainModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class EfficientNetB0ForHLabelCls(ExplainableMixInMMPretrainModel, MMPretrainHlabelClsModel):
    """EfficientNetB0 Model for hierarchical label classification task."""

    def __init__(
        self,
        hlabel_info: HLabelInfo,
        optimizer: OptimizerCallable = lambda params: torch.optim.SGD(
            params=params,
            lr=0.0049,
        ),
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = lambda optimizer: ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=1,
            monitor="val/accuracy",
        ),
        metric: MetricCallable = HLabelClsMetricCallble,
        torch_compile: bool = False,
    ) -> None:
        config = read_mmconfig(model_name="efficientnet_b0_light", subdir_name="hlabel_classification")

        super().__init__(
            hlabel_info=hlabel_info,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "hlabel", add_prefix)


class EfficientNetB0ForMulticlassCls(ExplainableMixInMMPretrainModel, MMPretrainMulticlassClsModel):
    """EfficientNetB0 Model for multi-label classification task."""

    def __init__(
        self,
        num_classes: int,
        light: bool = False,
        optimizer: OptimizerCallable = lambda params: torch.optim.SGD(
            params=params,
            lr=0.0049,
            momentum=0.9,
            weight_decay=0.0001,
        ),
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = lambda optimizer: ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=1,
            monitor="val/accuracy",
        ),
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        model_name = "efficientnet_b0_light" if light else "efficientnet_b0"
        config = read_mmconfig(model_name=model_name, subdir_name="multiclass_classification")
        super().__init__(
            num_classes=num_classes,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multiclass", add_prefix)

    def _reset_prediction_layer(self, num_classes: int) -> None:
        return

class EfficientNetB0ForMultilabelCls(ExplainableMixInMMPretrainModel, MMPretrainMultilabelClsModel):
    """EfficientNetB0 Model for multi-class classification task."""

    def __init__(
        self,
        num_classes: int,
        optimizer: OptimizerCallable = lambda params: torch.optim.SGD(
            params=params,
            lr=0.0049,
        ),
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = lambda optimizer: ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=1,
            monitor="val/accuracy",
        ),
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        config = read_mmconfig(model_name="efficientnet_b0_light", subdir_name="multilabel_classification")
        super().__init__(
            num_classes=num_classes,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multilabel", add_prefix)
