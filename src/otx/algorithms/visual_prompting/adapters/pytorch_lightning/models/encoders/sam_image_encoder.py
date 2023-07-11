"""Image encoder module for SAM."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from omegaconf import DictConfig
from torch import Tensor, nn

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.backbones.vit import (
    build_vit,
)


class SAMImageEncoder(nn.Module):
    """Image encoder module for SAM.

    Args:
        config (DictConfig): Config for image encoder.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        if "vit" in config.backbone:
            self.backbone = build_vit(config.backbone, config.image_size)
        else:
            raise NotImplementedError(
                (f"{config.backbone} for image encoder of SAM is not implemented yet. " f"Use vit_b, l, or h.")
            )

    def forward(self, images: Tensor) -> Tensor:
        """Forward function of image encoder.

        Args:
            images (Tensor): Input tensor.

        Returns:
            image_embeddings (Tensor): Output tensor.
        """
        image_embeddings = self.backbone(images)
        return image_embeddings
