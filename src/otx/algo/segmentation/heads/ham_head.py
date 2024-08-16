# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of HamburgerNet head."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
import torch.nn.functional as f
from torch import nn

from otx.algo.modules import Conv2dModule
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.norm import build_norm_layer
from otx.algo.segmentation.modules import resize

from .base_segm_head import BaseSegmHead

if TYPE_CHECKING:
    from pathlib import Path


class Hamburger(nn.Module):
    """Hamburger Module.

    It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
    """

    def __init__(
        self,
        ham_channels: int,
        ham_kwargs: dict[str, Any],
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Initialize Hamburger Module."""
        super().__init__()

        self.ham_in = Conv2dModule(ham_channels, ham_channels, 1, normalization=None, activation=None)

        self.ham = NMF2D(ham_channels=ham_channels, **ham_kwargs)

        self.ham_out = Conv2dModule(
            ham_channels,
            ham_channels,
            1,
            normalization=build_norm_layer(normalization, num_features=ham_channels),
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        enjoy = self.ham_in(x)
        enjoy = f.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)

        return f.relu(x + enjoy, inplace=True)


class NNLightHamHead(BaseSegmHead):
    """SegNeXt decode head."""

    def __init__(
        self,
        in_channels: int | list[int],
        channels: int,
        num_classes: int,
        dropout_ratio: float = 0.1,
        normalization: Callable[..., nn.Module] | None = partial(
            build_norm_layer,
            nn.GroupNorm,
            num_groups=32,
            requires_grad=True,
        ),
        activation: Callable[..., nn.Module] | None = nn.ReLU,
        in_index: int | list[int] = [1, 2, 3],  # noqa: B006
        input_transform: str | None = "multiple_select",
        align_corners: bool = False,
        pretrained_weights: Path | str | None = None,
        ham_channels: int = 512,
        ham_kwargs: dict[str, Any] | None = None,
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
            ham_kwargs (Dict[str, Any] | None): kwagrs for Ham.
                If None: {"md_r": 16, "md_s": 1, "eval_steps": 7, "train_steps": 6} will be used.

        Returns:
            None
        """
        super().__init__(
            input_transform=input_transform,
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            normalization=normalization,
            activation=activation,
            in_index=in_index,
            align_corners=align_corners,
            pretrained_weights=pretrained_weights,
        )

        if not isinstance(self.in_channels, list):
            msg = f"Input channels type must be list, but got {type(self.in_channels)}"
            raise TypeError(msg)

        self.ham_channels = ham_channels
        self.ham_kwargs = (
            ham_kwargs if ham_kwargs is not None else {"md_r": 16, "md_s": 1, "eval_steps": 7, "train_steps": 6}
        )

        self.squeeze = Conv2dModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            normalization=build_norm_layer(self.normalization, num_features=self.ham_channels),
            activation=build_activation_layer(self.activation),
        )

        self.hamburger = Hamburger(self.ham_channels, ham_kwargs=self.ham_kwargs, normalization=normalization)

        self.align = Conv2dModule(
            self.ham_channels,
            self.channels,
            1,
            normalization=build_norm_layer(self.normalization, num_features=self.channels),
            activation=build_activation_layer(self.activation),
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


class LightHamHead:
    """LightHamHead factory for segmentation."""

    HAMHEAD_CFG: ClassVar[dict[str, Any]] = {
        "segnext_base": {
            "in_channels": [128, 320, 512],
            "channels": 512,
            "ham_channels": 512,
        },
        "segnext_small": {
            "in_channels": [128, 320, 512],
            "channels": 256,
            "ham_channels": 256,
        },
        "segnext_tiny": {
            "in_channels": [64, 160, 256],
            "channels": 256,
            "ham_channels": 256,
        },
    }

    def __new__(cls, version: str, num_classes: int) -> NNLightHamHead:
        """Constructor for FCNHead."""
        if version not in cls.HAMHEAD_CFG:
            msg = f"model type '{version}' is not supported"
            raise KeyError(msg)

        return NNLightHamHead(**cls.HAMHEAD_CFG[version], num_classes=num_classes)
