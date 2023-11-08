"""Image encoder module for SAM."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig
from torch import Tensor, nn

from otx.v2.adapters.torch.lightning.modules.models.backbones import build_tiny_vit, build_vit


class SAMImageEncoder(nn.Module):
    """Image encoder module for SAM."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the SAM image encoder with the given configuration.

        Args:
            config (DictConfig): The configuration for the image encoder.

        Raises:
            NotImplementedError: If the specified backbone is not implemented yet.
        """
        super().__init__()
        if config.backbone == "tiny_vit":
            self.backbone = build_tiny_vit(config.image_size)
        elif "vit" in config.backbone:
            self.backbone = build_vit(config.backbone, config.image_size)
        else:
            msg = f"{config.backbone} for image encoder of SAM is not implemented yet. Use vit_b, l, or h."
            raise NotImplementedError(msg)

    def forward(self, images: Tensor) -> Tensor:
        """Forward function of image encoder.

        Args:
            images (Tensor): Input tensor.

        Returns:
            image_embeddings (Tensor): Output tensor.
        """
        return self.backbone(images)
