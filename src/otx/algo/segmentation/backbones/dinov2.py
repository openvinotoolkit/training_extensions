# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINO-V2 model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity
from otx.core.model.entity.classification import OTXClassificationModel

if TYPE_CHECKING:
    from omegaconf import DictConfig

from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
class DinoVisionTransformer(BaseModule):
    """DINO-v2 Model."""
    def __init__(
        self,
        backbone_name: str,
        freeze_backbone: bool,
        init_cfg: DictConfig | None = None,
    ):
        super().__init__(init_cfg)
        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone_name,
        )

        if freeze_backbone:
            self._freeze_backbone(self.backbone)

    def _freeze_backbone(self, backbone: nn.Module) -> None:
        """Freeze the backbone."""
        for _, v in backbone.named_parameters():
            v.requires_grad = False

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return self.backbone(imgs)
