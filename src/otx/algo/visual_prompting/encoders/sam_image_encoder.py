# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM image encoder model for the OTX visual prompting."""

from torch import nn


class SAMImageEncoder(nn.Module):
    backbone_configs = dict(
        tiny_vit=dict(
            img_size=1024,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            drop_path_rate=0.,
        )
    )
    
    def __new__(cls, backbone: str, *args, **kwargs):
        if backbone.lower() == "tiny_vit":
            from otx.algo.visual_prompting.backbones.tiny_vit import TinyViT as BACKBONE
        else:
            raise ValueError(f"{backbone} is not supported for SAMImageEncoder. Set among tiny_vit and vit_b.")
        return BACKBONE(**cls.backbone_configs.get(backbone))
