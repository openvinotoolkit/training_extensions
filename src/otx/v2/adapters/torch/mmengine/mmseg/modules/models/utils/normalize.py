"""Normalization."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Callable

import torch
import torch.nn.functional
from torch import nn


class OnnxLpNormalization(torch.autograd.Function):
    """OnnxLpNormalization."""

    @staticmethod
    def forward(
        ctx: Callable,  # noqa: ARG004
        x: torch.Tensor,
        axis: int = 0,
        p: int = 2,  # noqa: ARG004
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Forward."""
        denom = x.norm(2, axis, True).clamp_min(eps).expand_as(x)
        return x / denom

    @staticmethod
    def symbolic(
        g: torch.Graph,
        x: torch.Tensor,
        axis: int = 0,
        p: int = 2,
        eps: float = 1e-12,  # noqa: ARG004
    ) -> torch.Graph:
        """Symbolic onnxLpNormalization."""
        return g.op("LpNormalization", x, axis_i=int(axis), p_i=int(p))


def normalize(x: torch.Tensor, dim: int, p: int = 2, eps: float = 1e-12) -> torch.Tensor:
    """Normalize method."""
    if torch.onnx.is_in_onnx_export():
        return OnnxLpNormalization.apply(x, dim, p, eps)
    return torch.nn.functional.normalize(x, dim=dim, p=p, eps=eps)


class Normalize(nn.Module):
    """Normalizes a tensor along a given dimension."""

    def __init__(self, dim: int = 1, p: int = 2, eps: float = 1e-12) -> None:
        """Initializes Normalize.

        Args:
            dim (int): The dimension to normalize along.
            p (float): The exponent value in the norm formulation.
            eps (float): A small value to avoid division by zero.
        """
        super().__init__()

        self.dim = dim
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return normalize(x, self.dim, self.p, self.eps)
