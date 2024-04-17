# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of HamburgerNet head."""

import torch
import torch.nn.functional as f
from mmseg.models.decode_heads.ham_head import NMF2D, LightHamHead
# from mmseg.registry import MODELS

from .custom_fcn_head import ClassIncrementalMixin

import torch
import torch.nn as nn
import torch.nn.functional as F
from otx.algo.segmentation.blocks.blocks import ConvModule, resize


# class NMF2D(nn.Module):
#     """Non-negative Matrix Factorization (NMF) module.

#     It is modified version from mmsegmentation to avoid randomness in inference.
#     """

#     def __init__(self, spatial=True,
#                          S=1,
#                          R=64,
#                          train_steps=6,
#                          eval_steps=7,
#                          inv_t=1,
#                          eta=0.9,
#                          ham_channels: int = 512):
#         super().__init__()

#         self.spatial = spatial

#         self.S = S
#         self.R = R

#         self.train_steps = train_steps
#         self.eval_steps = eval_steps

#         self.inv_t = inv_t
#         self.eta = eta
#         bases = f.normalize(torch.rand((self.S, ham_channels // self.S, self.R)))
#         self.bases = torch.nn.parameter.Parameter(bases, requires_grad=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward Function."""
#         batch, channels, height, width = x.shape

#         # (B, C, H, W) -> (B * S, D, N)
#         scale = channels // self.S
#         x = x.view(batch * self.S, scale, height * width)

#         # (S, D, R) -> (B * S, D, R)
#         if self.training:
#             bases = self._build_bases(batch, self.S, scale, self.R, device=x.device)
#         else:
#             bases = self.bases.repeat(batch, 1, 1)

#         bases, coef = self.local_inference(x, bases)

#         # (B * S, N, R)
#         coef = self.compute_coef(x, bases, coef)

#         # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
#         x = torch.bmm(bases, coef.transpose(1, 2))

#         # (B * S, D, N) -> (B, C, H, W)
#         return x.view(batch, channels, height, width)
#     def _build_bases(self, B, S, D, R, device=None):
#         """Build bases in initialization."""
#         if device is None:
#             device = 'cpu'
#         bases = torch.rand((B * S, D, R)).to(device)
#         bases = F.normalize(bases, dim=1)

#         return bases

#     def local_step(self, x, bases, coef):
#         """Local step in iteration to renew bases and coefficient."""
#         # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
#         numerator = torch.bmm(x.transpose(1, 2), bases)
#         # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
#         denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
#         # Multiplicative Update
#         coef = coef * numerator / (denominator + 1e-6)

#         # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
#         numerator = torch.bmm(x, coef)
#         # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
#         denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
#         # Multiplicative Update
#         bases = bases * numerator / (denominator + 1e-6)

#         return bases, coef

#     def local_inference(self, x, bases):
#         # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
#         coef = torch.bmm(x.transpose(1, 2), bases)
#         coef = F.softmax(self.inv_t * coef, dim=-1)

#         steps = self.train_steps if self.training else self.eval_steps
#         for _ in range(steps):
#             bases, coef = self.local_step(x, bases, coef)

#         return bases, coef

#     def compute_coef(self, x, bases, coef):
#         """Compute coefficient."""
#         # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
#         numerator = torch.bmm(x.transpose(1, 2), bases)
#         # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
#         denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
#         # multiplication update
#         coef = coef * numerator / (denominator + 1e-6)

#         return coef


# class Hamburger(nn.Module):
#     def __init__(self,
#                  ham_channels=512,
#                  spatial=True,
#                  MD_S=1,   #MD_S
#                  MD_R=64,  #MD_R
#                  train_steps=6,
#                  eval_steps=7,
#                  inv_t=100,
#                  eta=0.9):
#         super().__init__()

#         '''
#         self.ham_in = ConvModule(
#             ham_channels,
#             ham_channels,
#             1,
#             norm_cfg=None,
#             act_cfg=None
#         )

#         self.ham = NMF2D(ham_kwargs)

#         self.ham_out = ConvModule(
#             ham_channels,
#             ham_channels,
#             1,
#             norm_cfg=norm_cfg,
#             act_cfg=None)
#         '''

#         # add Relu at end as NMF works of non-negative only
#         self.ham_in = ConvModule(ham_channels, ham_channels, 1, bias=True)
#         self.ham = NMF2D(spatial, MD_S, MD_R, train_steps, eval_steps, inv_t, eta)
#         self.ham_out = ConvModule(ham_channels, ham_channels, 1, bias=False, norm_cfg=dict(type='GN', num_groups=32))


#     def forward(self, x):
#         enjoy = self.ham_in(x)
#         enjoy = F.relu(enjoy, inplace=True)
#         enjoy = self.ham(enjoy)
#         enjoy = self.ham_out(enjoy)
#         ham = F.relu(x + enjoy, inplace=True)

#         return ham


# class LightHamHead(ClassIncrementalMixin, nn.Module):
#     """Is Attention Better Than Matrix Decomposition?
#     This head is the implementation of `HamNet
#     <https://arxiv.org/abs/2109.04553>`_.
#     Args:
#         ham_channels (int): input channels for Hamburger.
#         ham_kwargs (int): kwagrs for Ham.

#     """

#     '''SegNext'''
#     def __init__(self, in_channels=[128,320,512], ham_channels=512, channels=512,
#                  spatial=True,
#                  MD_S=1,   #MD_S
#                  MD_R=64,  #MD_R
#                  train_steps=6,
#                  eval_steps=7,
#                  inv_t=100,
#                  eta=0.9):
#         super().__init__()

#         self.squeeze = ConvModule(sum(in_channels), ham_channels, 1, bias=False, norm_cfg=dict(type='GN', num_groups=32), act_cfg='ReLU')
#         self.hamburger = Hamburger(ham_channels, spatial, MD_S, MD_R, train_steps, eval_steps, inv_t, eta)
#         self.align = ConvModule(ham_channels, channels, 1, bias=False, norm_cfg=dict(type='GN', num_groups=32), act_cfg='ReLU')

#     def forward(self, inputs):

#         inputs = inputs[1:] # drop stage 1 features b/c low level
#         inputs = [resize(level, size=inputs[-3].shape[2:], mode='bilinear') for level in inputs]

#         x = torch.cat(inputs, dim=1)
#         x = self.squeeze(x)

#         x = self.hamburger(x)

#         x = self.align(x)

#         return x

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
