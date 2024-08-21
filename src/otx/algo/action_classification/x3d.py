# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""X3D model implementation."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from torch import nn

from otx.algo.action_classification.backbones.x3d import X3DBackbone
from otx.algo.action_classification.heads.x3d_head import X3DHead
from otx.algo.action_classification.recognizers.recognizer import BaseRecognizer
from otx.algo.modules.norm import build_norm_layer
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


class X3D(OTXActionClsModel):
    """X3D Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (224, 224),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.load_from = "https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth"
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.mean = (114.75, 114.75, 114.75)
        self.std = (57.38, 57.38, 57.38)

    def _create_model(self) -> nn.Module:
        model = self._build_model(num_classes=self.label_info.num_classes)
        model.init_weights()
        self.classification_layers = self.get_classification_layers(prefix="model.")

        if self.load_from is not None:
            load_checkpoint(model, self.load_from, map_location="cpu")

        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        return BaseRecognizer(
            backbone=X3DBackbone(
                gamma_b=2.25,
                gamma_d=2.2,
                gamma_w=1,
                normalization=partial(build_norm_layer, nn.BatchNorm3d, requires_grad=True),
                activation=partial(nn.ReLU, inplace=True),
            ),
            cls_head=X3DHead(
                num_classes=num_classes,
                in_channels=432,
                hidden_dim=2048,
                loss_cls=nn.CrossEntropyLoss(),
                spatial_type="avg",
                dropout_ratio=0.5,
                average_clips="prob",
                fc1_bias=False,
            ),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_action_ckpt(state_dict, add_prefix)
