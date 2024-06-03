# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""X3D model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.algo.action_classification.backbones.movinet import OTXMoViNet as MoViNetBackbone
from otx.algo.action_classification.heads.movinet_head import MoViNetHead
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.metrics.accuracy import MultiClassClsMetricCallable
from otx.core.model.action_classification import OTXActionClsModel
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.core.metrics import MetricCallable


class MoViNet(OTXActionClsModel):
    """MoViNet Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def get_classification_layers(self, prefix: str = "model.") -> dict[str, dict[str, int]]:
        """Get final classification layer information for incremental learning case."""
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                incremental_model_dim = incremental_model_dict[key].shape[0]
                stride = incremental_model_dim - sample_model_dim
                num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
                classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
        return classification_layers

    def _create_model(self) -> nn.Module:
        detector = self._build_model(num_classes=self.label_info.num_classes)
        detector.init_weights()
        self.classification_layers = self.get_classification_layers(prefix="model.")

        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")

        return detector

    def _build_model(self, num_classes: int) -> nn.Module:
        return ImageClassifier(
            backbone=MoViNetBackbone(version=self.version, pretrained=True),
            head=MoViNetHead(
                num_classes=num_classes,
                in_channels=480,
                hidden_dim=2048,
                topk=(1, 5) if num_classes >= 5 else (1,),
                loss=nn.CrossEntropyLoss(reduction="none"),
            ),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_action_ckpt(state_dict, add_prefix)
