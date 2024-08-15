# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM image encoder model for the OTX visual prompting."""

from typing import ClassVar

from torch import nn


class SAMImageEncoder(nn.Module):
    """Image encoder model of Segment Anything for visual prompting model."""

    backbone_configs: ClassVar[dict] = {
        "tiny_vit": {
            "embed_dims": [64, 128, 160, 320],
            "depths": [2, 2, 6, 2],
            "num_heads": [2, 4, 5, 10],
            "window_sizes": [7, 7, 14, 7],
        },
        "vit_b": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "global_attn_indexes": [2, 5, 8, 11],
        },
        "vit_l": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "global_attn_indexes": [5, 11, 17, 23],
        },
        "vit_h": {
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "global_attn_indexes": [7, 15, 23, 31],
        },
    }

    def __new__(cls, backbone_type: str, *args, **kwargs):  # noqa: ARG003
        """Initialize image encoder to target backbone."""
        if backbone_type.lower() == "tiny_vit":
            from otx.algo.visual_prompting.backbones.tiny_vit import TinyViT

            return TinyViT(**{**cls.backbone_configs.get(backbone_type.lower()), **kwargs})  # type: ignore[dict-item]
        elif backbone_type.lower() in ["vit_b", "vit_l", "vit_h"]:  # noqa: RET505
            from otx.algo.visual_prompting.backbones.vit import ViT

            return ViT(**{**cls.backbone_configs.get(backbone_type.lower()), **kwargs})  # type: ignore[dict-item]

        else:
            error_log = f"{backbone_type} is not supported for SAMImageEncoder. Set among tiny_vit and vit_b."
            raise ValueError(error_log)
