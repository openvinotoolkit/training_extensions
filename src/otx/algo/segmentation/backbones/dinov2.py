# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINO-V2 model for the OTX classification."""

from __future__ import annotations
from typing import Dict
from functools import partial

import torch
from mmengine.model import BaseModule
from mmseg.models.builder import BACKBONES
from torch import nn


@BACKBONES.register_module()
class DinoVisionTransformer(BaseModule):
    """DINO-v2 Model."""

    def __init__(
        self,
        name: str,
        freeze_backbone: bool,
        out_index: list[int],
<<<<<<< HEAD
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg)
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # noqa: SLF001, ARG005
=======
        init_cfg: Dict | None = None
    ):
        super().__init__(init_cfg)
        torch.hub._validate_not_a_forked_repo=lambda a, b, c: True # noqa: SLF001, ARG005
>>>>>>> eb8948680 (fixed issues. Adapt hyperparametrs)
        self.backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=name)
        if freeze_backbone:
            self._freeze_backbone(self.backbone)

        # take intermediate layers to preserve spatial dimension
        self.backbone.forward = partial(
            self.backbone.get_intermediate_layers,
            n=out_index,
            reshape=True,
        )

    def _freeze_backbone(self, backbone: nn.Module) -> None:
        """Freeze the backbone."""
        for _, v in backbone.named_parameters():
            v.requires_grad = False

<<<<<<< HEAD
    def init_weights(self) -> None:
        """Initialize the weights."""
=======
    def init_weights(self):
>>>>>>> eb8948680 (fixed issues. Adapt hyperparametrs)
        # restrict rewriting backbone pretrained weights from torch.hub
        # unless weights passed explicitly in config
        if self.init_cfg:
            return super().init_weights()
<<<<<<< HEAD
        return None
=======
>>>>>>> eb8948680 (fixed issues. Adapt hyperparametrs)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return self.backbone(imgs)
