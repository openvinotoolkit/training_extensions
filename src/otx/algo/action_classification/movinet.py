# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""X3D model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from otx.algo.action_classification.backbones.movinet import MoViNetBackbone
from otx.algo.action_classification.heads.movinet_head import MoViNetHead
from otx.algo.action_classification.recognizers.movinet_recognizer import MoViNetRecognizer
from otx.algo.utils.mmengine_utils import load_checkpoint
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.metrics.accuracy import MultiClassClsMetricCallable
from otx.core.model.action_classification import OTXActionClsModel
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class MoViNet(OTXActionClsModel):
    """MoViNet Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (224, 224),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.load_from = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA0_statedict_v3?raw=true"
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        model = self._build_model(num_classes=self.label_info.num_classes)
        model.init_weights()
        self.classification_layers = self.get_classification_layers(prefix="model.")

        if self.load_from is not None:
            load_checkpoint(model, self.load_from, map_location="cpu")

        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        return MoViNetRecognizer(
            backbone=MoViNetBackbone(),
            cls_head=MoViNetHead(
                num_classes=num_classes,
                in_channels=480,
                hidden_dim=2048,
                loss_cls=nn.CrossEntropyLoss(),
            ),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_action_ckpt(state_dict, add_prefix)
