# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of HamburgerNet head."""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as f
from mmengine.device import get_device
from torch import nn

from otx.algo.modules import ConvModule
from otx.algo.segmentation.modules import resize

from .base_head import BaseSegmHead


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(
        self,
        ham_channels: int = 512,
        ham_kwargs: Dict[str, Any] = dict(),
        norm_cfg: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Hamburger Module.

        Args:
            ham_channels (int): Input and output channels of feature.
            ham_kwargs (Dict[str, Any]): Config of matrix decomposition module.
            norm_cfg (Optional[Dict[str, Any]]): Config of norm layers.
        """
        super().__init__()

        self.ham_in = ConvModule(ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_channels=ham_channels, **ham_kwargs)

        self.ham_out = ConvModule(ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = f.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = f.relu(x + enjoy, inplace=True)

        return ham


class LightHamHead(BaseSegmHead):
    """SegNeXt decode head."""

    def __init__(
        self,
        ham_channels: int = 512,
        ham_kwargs: Dict[str, Any] = dict(),
        **kwargs: Any,
    ) -> None:
        """SegNeXt decode head.

        This decode head is the implementation of `SegNeXt: Rethinking
        Convolutional Attention Design for Semantic
        Segmentation <https://arxiv.org/abs/2209.08575>`_.
        Inspiration from https://github.com/visual-attention-network/segnext.

        Specifically, LightHamHead is inspired by HamNet from
        `Is Attention Better Than Matrix Decomposition?
        <https://arxiv.org/abs/2109.04553>`.

        Args:
            ham_channels (int): input channels for Hamburger.
                Defaults to 512.
            ham_kwargs (Dict[str, Any]): kwagrs for Ham. Defaults to an empty dictionary.

        Returns:
            None
        """
        super().__init__(input_transform="multiple_select", **kwargs)
        if not isinstance(self.in_channels, list):
            msg = f"Input channels type must be list, but got {type(self.in_channels)}"
            raise TypeError(msg)

        self.ham_channels: int = ham_channels
        self.ham_kwargs: Dict[str, Any] = ham_kwargs

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.hamburger = Hamburger(self.ham_channels, self.ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [
            resize(level, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)  # type: ignore[assignment]
            for level in inputs  # type: ignore[assignment]
        ]

        inputs = torch.cat(inputs, dim=1)
        # apply a conv block to squeeze feature map
        x = self.squeeze(inputs)  # type: ignore[has-type]
        # apply hamburger module
        x = self.hamburger(x)  # type: ignore[has-type]

        # apply a conv block to align feature map
        output = self.align(x)  # type: ignore[has-type]
        output = self.cls_seg(output)
        return output


class NMF2D(nn.Module):
    """Non-negative Matrix Factorization (NMF) module.

    It is modified version from mmsegmentation to avoid randomness in inference.
    """

    def __init__(
        self,
        ham_channels: int = 512,
        MD_S=1,
        MD_R=64,
        train_steps=6,
        eval_steps=7,
        inv_t=1,
        rand_init=True,
    ):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.rand_init = rand_init
        bases = f.normalize(torch.rand((self.S, ham_channels // self.S, self.R)))
        self.bases = torch.nn.parameter.Parameter(bases, requires_grad=False)
        self.inv_t = 1

    def local_inference(self, x: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
        """Local inference."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = f.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

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

    def _build_bases(self, B: int, S: int, D: int, R: int, device: torch.device | None = None) -> torch.Tensor:
        """Build bases in initialization.

        Args:
            B (int): Batch size.
            S (int): Number of segmentations.
            D (int): Number of input channels.
            R (int): Number of basis vectors.
            device (Optional[torch.device]): Device to place the tensor on. Defaults to None.

        Returns:
            torch.Tensor: Tensor of shape (B * S, D, R) containing the built bases.
        """
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = f.normalize(bases, dim=1)

        return bases

    def local_step(self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Local step in iteration to renew bases and coefficient.

        Args:
            x (torch.Tensor): Input tensor of shape (B * S, D, N).
            bases (torch.Tensor): Basis tensor of shape (B * S, D, R).
            coef (torch.Tensor): Coefficient tensor of shape (B * S, N, R).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Renewed bases and coefficients.
        """
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        """Compute coefficient.

        Args:
            x (torch.Tensor): Input tensor of shape (B * S, D, N).
            bases (torch.Tensor): Basis tensor of shape (B * S, D, R).
            coef (torch.Tensor): Coefficient tensor of shape (B * S, N, R).

        Returns:
            torch.Tensor: Tensor of shape (B * S, N, R) containing the computed coefficients.
        """
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef
