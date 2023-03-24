"""Normalization."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
from torch import nn


class OnnxLpNormalization(torch.autograd.Function):  # pylint: disable=abstract-method
    """OnnxLpNormalization."""

    @staticmethod
    def forward(ctx, x, axis=0, p=2, eps=1e-12):  # pylint: disable=unused-argument
        """Forward."""
        denom = x.norm(2, axis, True).clamp_min(eps).expand_as(x)
        return x / denom

    @staticmethod
    def symbolic(g, x, axis=0, p=2, eps=1e-12):  # pylint: disable=invalid-name, unused-argument
        """Symbolic onnxLpNormalization."""
        return g.op("LpNormalization", x, axis_i=int(axis), p_i=int(p))


def normalize(x, dim, p=2, eps=1e-12):
    """Normalize method."""
    if torch.onnx.is_in_onnx_export():
        return OnnxLpNormalization.apply(x, dim, p, eps)
    return F.normalize(x, dim=dim, p=p, eps=eps)


class Normalize(nn.Module):
    """Normalize."""

    def __init__(self, dim=1, p=2, eps=1e-12):
        super().__init__()

        self.dim = dim
        self.p = p
        self.eps = eps

    def forward(self, x):
        """Forward."""
        return normalize(x, self.dim, self.p, self.eps)
