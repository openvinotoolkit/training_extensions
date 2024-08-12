# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of HamburgerNet head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as f
from torch import nn

from otx.algo.modules import Conv2dModule
from otx.algo.segmentation.modules import resize

from .base_segm_head import BaseSegmHead


class Hamburger(nn.Module):
    """Hamburger Module.

    It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(
        self,
        ham_channels: int,
        ham_kwargs: dict[str, Any],
        norm_cfg: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize Hamburger Module.

        Args:
            ham_channels (int): Input and output channels of feature.
            ham_kwargs (Dict[str, Any]): Config of matrix decomposition module.
            norm_cfg (Optional[Dict[str, Any]]): Config of norm layers.
        """
        super().__init__()

        self.ham_in = Conv2dModule(ham_channels, ham_channels, 1, norm_cfg=None, activation_callable=None)

        self.ham = NMF2D(ham_channels=ham_channels, **ham_kwargs)

        self.ham_out = Conv2dModule(ham_channels, ham_channels, 1, norm_cfg=norm_cfg, activation_callable=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        enjoy = self.ham_in(x)
        enjoy = f.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)

        return f.relu(x + enjoy, inplace=True)


class LightHamHead(BaseSegmHead):
    """SegNeXt decode head."""

    def __init__(
        self,
        ham_channels: int = 512,
        ham_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
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
        self.ham_kwargs: dict[str, Any] = ham_kwargs if ham_kwargs is not None else {}

        self.squeeze = Conv2dModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            norm_cfg=self.norm_cfg,
            activation_callable=self.activation_callable,
        )

        self.hamburger = Hamburger(self.ham_channels, ham_kwargs=self.ham_kwargs, **kwargs)

        self.align = Conv2dModule(
            self.ham_channels,
            self.channels,
            1,
            norm_cfg=self.norm_cfg,
            activation_callable=self.activation_callable,
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
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
        return self.cls_seg(output)


class NMF2D(nn.Module):
    """Non-negative Matrix Factorization (NMF) module.

    It is modified version from mmsegmentation to avoid randomness in inference.
    """

    def __init__(
        self,
        ham_channels: int,
        md_s: int = 1,
        md_r: int = 64,
        train_steps: int = 6,
        eval_steps: int = 7,
        inv_t: int = 1,
        rand_init: bool = True,
    ) -> None:
        """Initialize Non-negative Matrix Factorization (NMF) module.

        Args:
            ham_channels (int): Number of input channels.
            md_s (int): Number of spatial coefficients in Matrix Decomposition.
            md_r (int): Number of latent dimensions R in Matrix Decomposition.
            train_steps (int): Number of iteration steps in Multiplicative Update (MU)
                rule to solve Non-negative Matrix Factorization (NMF) in training.
            eval_steps (int): Number of iteration steps in Multiplicative Update (MU)
                rule to solve Non-negative Matrix Factorization (NMF) in evaluation.
            inv_t (int): Inverted multiple number to make coefficient smaller in softmax.
            rand_init (bool): Whether to initialize randomly.
        """
        super().__init__()

        self.s = md_s
        self.r = md_r

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.rand_init = rand_init
        bases = f.normalize(torch.rand((self.s, ham_channels // self.s, self.r)))
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
        scale = channels // self.s
        x = x.view(batch * self.s, scale, height * width)

        # (S, D, R) -> (B * S, D, R)
        if self.training:
            bases = self._build_bases(batch, self.s, scale, self.r, device=x.device)
        else:
            bases = self.bases.repeat(batch, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        return x.view(batch, channels, height, width)

    def _build_bases(
        self,
        batch_size: int,
        segments: int,
        channels: int,
        basis_vectors: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build bases in initialization.

        Args:
            batch_size (int): Batch size.
            segments (int): Number of segmentations.
            channels (int): Number of input channels.
            basis_vectors (int): Number of basis vectors.
            device (Optional[torch.device]): Device to place the tensor on. Defaults to None.

        Returns:
            torch.Tensor: Tensor of shape (batch_size * segments, channels, basis_vectors) containing the built bases.
        """
        bases = torch.rand((batch_size * segments, channels, basis_vectors)).to(device)

        return f.normalize(bases, dim=1)

    def local_step(self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        return coef * numerator / (denominator + 1e-6)
