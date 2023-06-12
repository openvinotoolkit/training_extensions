"""Image encoder module for SAM."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
from omegaconf import DictConfig

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.backbone import (
    build_vit,
)


class SAMImageEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        if "vit" in config.image_encoder.backbone.name:
            self.backbone = build_vit(config)
        else:
            raise NotImplementedError((
                f"{config.image_encoder.backbone.name} for image encoder of SAM is not implemented yet. "
                f"Use ViT-B, L, or H."
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
