# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DINO-V2 model for the OTX classification."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch import nn

from otx.algo.modules.base_module import BaseModule
from otx.algo.utils.mmengine_utils import load_checkpoint_to_model, load_from_http
from otx.utils.utils import get_class_initial_arguments


class DinoVisionTransformer(BaseModule):
    """DINO-v2 Model."""

    def __init__(
        self,
        name: str,
        freeze_backbone: bool,
        out_index: list[int],
        init_cfg: dict | None = None,
        pretrained_weights: str | None = None,
    ):
        super().__init__(init_cfg)
        self._init_args = get_class_initial_arguments()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # noqa: SLF001, ARG005
        self.backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=name)
        if freeze_backbone:
            self._freeze_backbone(self.backbone)

        # take intermediate layers to preserve spatial dimension
        self.backbone.forward = partial(
            self.backbone.get_intermediate_layers,
            n=out_index,
            reshape=True,
        )

        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def _freeze_backbone(self, backbone: nn.Module) -> None:
        """Freeze the backbone."""
        for _, v in backbone.named_parameters():
            v.requires_grad = False

    def init_weights(self) -> None:
        """Initialize the weights."""
        # restrict rewriting backbone pretrained weights from torch.hub
        # unless weights passed explicitly in config
        if self.init_cfg:
            return super().init_weights()
        return None

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return self.backbone(imgs)

    def load_pretrained_weights(self, pretrained: str | None = None, prefix: str = "") -> None:
        """Initialize weights."""
        checkpoint = None
        if isinstance(pretrained, str) and Path(pretrained).exists():
            checkpoint = torch.load(pretrained, "cpu")
            print(f"init weight - {pretrained}")
        elif pretrained is not None:
            checkpoint = load_from_http(pretrained, "cpu")
            print(f"init weight - {pretrained}")
        if checkpoint is not None:
            load_checkpoint_to_model(self, checkpoint, prefix=prefix)

    def __reduce__(self):
        return (DinoVisionTransformer, self._init_args)
