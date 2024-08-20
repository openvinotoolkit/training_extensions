# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks/modules for semantic segmentation."""

from __future__ import annotations

from typing import Callable

import torch


class OnnxLpNormalization(torch.autograd.Function):
    """OnnxLpNormalization."""

    @staticmethod
    def forward(
        ctx: Callable,
        x: torch.Tensor,
        axis: int = 0,
        p: int = 2,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Forward."""
        del ctx, p  # These args are not used.
        denom = x.norm(2, axis, True).clamp_min(eps).expand_as(x)
        return x / denom

    @staticmethod
    def symbolic(
        g: torch.Graph,
        x: torch.Tensor,
        axis: int = 0,
        p: int = 2,
        eps: float = 1e-12,
    ) -> torch.Graph:
        """Symbolic onnxLpNormalization."""
        del eps  # These args are not used.
        return g.op("LpNormalization", x, axis_i=int(axis), p_i=int(p))
