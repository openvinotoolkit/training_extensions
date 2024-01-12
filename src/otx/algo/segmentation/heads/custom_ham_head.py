"""Implementation of HamburgerNet head."""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from mmseg.models.decode_heads.ham_head import NMF2D, LightHamHead
from mmseg.registry import MODELS


class CustomNMF2D(NMF2D):
    """Non-negative Matrix Factorization (NMF) module.

    It is modified version from mmsegmentation to avoid randomness in inference.
    """

    def __init__(self, ham_channels=512, **kwargs):
        super().__init__(kwargs)
        bases = F.normalize(torch.rand((self.S, ham_channels // self.S, self.R)))
        self.bases = torch.nn.parameter.Parameter(bases)

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        bases = self.bases.repeat(B, 1, 1)
        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


@MODELS.register_module()
class CustomLightHamHead(LightHamHead):
    """SegNeXt decode head.

    It is modified LightHamHead from mmsegmentation.

    Args:
        ham_channels (int): input channels for Hamburger.
            Defaults: 512.
        ham_kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super().__init__(ham_channels=ham_channels, ham_kwargs=ham_kwargs, **kwargs)

        self.hamburger.ham = CustomNMF2D(ham_channels=ham_channels, **ham_kwargs)
