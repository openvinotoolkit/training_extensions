"""Image encoder module for SAM."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from omegaconf import DictConfig
from torch import nn

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.backbones import (
    build_tiny_vit,
    build_vit,
)


class SAMImageEncoder(nn.Module):
    """Image encoder module for SAM.

    Args:
        config (DictConfig): Config for image encoder.
    """

    def __new__(cls, config: DictConfig):
        """Initialize SAM image encoder to the target backbone."""
        if "tiny_vit" == config.backbone:
            return build_tiny_vit(config.image_size)
        elif "vit" in config.backbone:
            return build_vit(config.backbone, config.image_size)
        else:
            raise NotImplementedError(
                (f"{config.backbone} for image encoder of SAM is not implemented yet. " f"Use vit_b, l, or h.")
            )
