# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of HamburgerNet head."""

import torch
import torch.nn.functional as f
from mmseg.models.decode_heads.ham_head import NMF2D, LightHamHead
from mmseg.registry import MODELS

from .custom_fcn_head import ClassIncrementalMixin


class CustomNMF2D(NMF2D):
    """Non-negative Matrix Factorization (NMF) module.

    It is modified version from mmsegmentation to avoid randomness in inference.
    """

    def __init__(self, ham_channels: int = 512, **kwargs):
        super().__init__(kwargs)
        bases = f.normalize(torch.rand((self.S, ham_channels // self.S, self.R)))
        self.bases = torch.nn.parameter.Parameter(bases, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Function."""
        batch, channels, height, width = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        scale = channels // self.S
        x = x.view(batch * self.S, scale, height * width)

        # (S, D, R) -> (B * S, D, R)
        if self.training:
            bases = self._build_bases(batch, self.S, scale, self.R, device=x.device)
        else:
            bases = self.bases.repeat(batch, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        return x.view(batch, channels, height, width)


@MODELS.register_module()
class CustomLightHamHead(ClassIncrementalMixin, LightHamHead):
    """SegNeXt decode head.

    It is modified LightHamHead from mmsegmentation.

    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (dict): kwagrs for Ham.
    """

    def __init__(self, ham_channels: int, ham_kwargs: dict, **kwargs):
        super().__init__(ham_channels=ham_channels, ham_kwargs=ham_kwargs, **kwargs)

        self.hamburger.ham = CustomNMF2D(ham_channels=ham_channels, **ham_kwargs)
