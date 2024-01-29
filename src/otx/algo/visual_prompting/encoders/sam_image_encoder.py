# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM image encoder model for the OTX visual prompting."""

from typing import ClassVar

from torch import nn
from functools import partial


class SAMImageEncoder(nn.Module):
    """Image encoder model of Segment Anything for visual prompting model."""

    backbone_configs: ClassVar[dict] = {
        "tiny_vit": {
            "img_size": 1024,
            "embed_dims": [64, 128, 160, 320],
            "depths": [2, 2, 6, 2],
            "num_heads": [2, 4, 5, 10],
            "window_sizes": [7, 7, 14, 7],
            "drop_path_rate": 0.0,
        },
        "vit_b": {
            "img_size": 1024,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "out_chans": 256,
            "patch_size": 16,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "use_rel_pos": True,
            "window_size": 14,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "global_attn_indexes": [2, 5, 8, 11],
        },
        "vit_l": {
            "img_size": 1024,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "out_chans": 256,
            "patch_size": 16,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "use_rel_pos": True,
            "window_size": 14,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "global_attn_indexes": [5, 11, 17, 23],
        },
        "vit_h": {
            "img_size": 1024,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "out_chans": 256,
            "patch_size": 16,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "use_rel_pos": True,
            "window_size": 14,
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "global_attn_indexes": [7, 15, 23, 31],
        },
    }

    def __new__(cls, backbone: str, *args, **kwargs):  # noqa: ARG003
        """Initialize image encoder to target backbone."""
        if backbone.lower() == "tiny_vit":
            from otx.algo.visual_prompting.backbones.tiny_vit import TinyViT

            return TinyViT(**cls.backbone_configs.get(backbone.lower()))  # type: ignore[arg-type]
        elif backbone.lower() in ["vit_b", "vit_l", "vit_h"]:
            from otx.algo.visual_prompting.backbones.vit import ViT
            
            return ViT(**cls.backbone_configs.get(backbone.lower()))
            
        else:  # noqa: RET505
            error_log = f"{backbone} is not supported for SAMImageEncoder. Set among tiny_vit and vit_b."
            raise ValueError(error_log)
