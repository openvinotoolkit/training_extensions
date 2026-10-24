# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared Vision Transformer building blocks (Attention, Block).

Unifies the near-identical ``Attention``/``Block`` implementations that previously
existed separately in ``detection/backbones/ecvit.py`` (EdgeCrafter) and
``detection/backbones/vit_tiny.py`` (DEIMv2). Each caller keeps its own MLP
implementation (``Mlp`` / ``MLP2L``) and injects it into :class:`Block` via the
``mlp_layer`` constructor parameter, so this unification does not change either
model family's MLP dropout-placement behavior.

Modified from DINOv3 (https://github.com/facebookresearch/dinov3).
Modified from https://huggingface.co/spaces/Hila/RobustViT/blob/main/ViT/ViT_new.py
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from getitune.backend.lightning.models.common.layers.transformer_layers import MLP2L
from getitune.backend.lightning.models.modules.drop import DropPath

__all__ = ["Attention", "Block"]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class Attention(nn.Module):
    """Multi-head self-attention with optional 2-D RoPE.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads. Defaults to 8.
        qkv_bias: Whether to add bias to QKV projection. Defaults to False.
        attn_drop: Attention dropout rate. Defaults to 0.0.
        proj_drop: Output projection dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, rope_sincos: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """Forward pass."""
        b, n, c = x.shape
        head_dim = c // self.num_heads
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if rope_sincos is not None:
            sin, cos = rope_sincos
            # register/cls token is at index 0; apply RoPE only to patch tokens
            q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]
            q_patch = _apply_rope(q_patch, sin, cos)
            k_patch = _apply_rope(k_patch, sin, cos)
            q = torch.cat((q_cls, q_patch), dim=2)
            k = torch.cat((k_cls, k_patch), dim=2)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """ViT transformer block with pre-normalization.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
        qkv_bias: Whether to add bias to QKV projection. Defaults to False.
        drop: Dropout rate for MLP. Defaults to 0.0.
        attn_drop: Attention dropout rate. Defaults to 0.0.
        drop_path: Drop path (stochastic depth) rate. Defaults to 0.0.
        act_layer: Activation layer class. Defaults to nn.GELU.
        norm_layer: Normalization layer class. Defaults to nn.LayerNorm.
        mlp_layer: MLP implementation, constructed as
            ``mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio),
            out_features=dim, act_layer=act_layer, drop=drop)``. Callers with
            different MLP dropout-placement semantics (e.g. EdgeCrafter's
            single-dropout ``Mlp`` vs. DEIMv2's double-dropout ``MLP2L``) should
            pass their own class here rather than relying on the default.
            Defaults to :class:`~getitune.backend.lightning.models.common.layers.transformer_layers.MLP2L`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] | Callable[..., nn.Module] = nn.LayerNorm,
        mlp_layer: Callable[..., nn.Module] = MLP2L,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: Tensor, rope_sincos: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """Forward pass."""
        x = x + self.drop_path(self.attn(self.norm1(x), rope_sincos=rope_sincos))
        return x + self.drop_path(self.mlp(self.norm2(x)))
