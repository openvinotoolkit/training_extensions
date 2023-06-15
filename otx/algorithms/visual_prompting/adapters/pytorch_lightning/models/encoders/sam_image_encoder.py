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
        if "vit" in config.backbone:
            self.backbone = build_vit(config.backbone, config.image_size)
        else:
            raise NotImplementedError((
                f"{config.backbone} for image encoder of SAM is not implemented yet. "
                f"Use vit_b, l, or h."
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
