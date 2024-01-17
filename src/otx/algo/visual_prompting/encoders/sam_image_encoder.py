# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM image encoder model for the OTX visual prompting."""

from typing import ClassVar

from torch import nn


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
    }

    def __new__(cls, backbone: str, *args, **kwargs):  # noqa: ARG003
        """Initialize image encoder to target backbone."""
        if backbone.lower() == "tiny_vit":
            from otx.algo.visual_prompting.backbones.tiny_vit import TinyViT

            return TinyViT(**cls.backbone_configs.get(backbone))  # type: ignore[arg-type]
        else:  # noqa: RET505
            error_log = f"{backbone} is not supported for SAMImageEncoder. Set among tiny_vit and vit_b."
            raise ValueError(error_log)
