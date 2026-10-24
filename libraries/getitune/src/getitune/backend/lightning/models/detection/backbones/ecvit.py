# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EC-ViT backbone (ViTAdapter) for EdgeCrafter models.

Modified from EdgeCrafter (https://github.com/Intellindust-AI-Lab/EdgeCrafter).
Modified from https://huggingface.co/spaces/Hila/RobustViT/blob/main/ViT/ViT_new.py

This implementation is licensed under Apache-2.0. Pretrained ECViT weights are
distilled from DINOv3 and are subject to the DINOv3 license.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, ClassVar

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from getitune.backend.lightning.models.common.layers.position_embed import RopePositionEmbedding
from getitune.backend.lightning.models.common.layers.vit_blocks import Block
from getitune.backend.lightning.models.detection.necks.dfine_hybrid_encoder import ConvNormLayerFusable
from getitune.backend.lightning.models.utils.weight_init import trunc_normal_

__all__ = ["ECViTAdapter"]


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class ConvPyramidPatchEmbed(nn.Module):
    """Convolutional pyramid patch embedding (patch_size=16).

    Replaces a single strided convolution with a chain of 3 x 3 stride-2 convs
    followed by a final 3 x 3 stride-2 projection. Supports Conv-BN fusion
    via ``ConvNormLayerFusable``.

    Args:
        embed_dim: Output embedding dimension.
        patch_size: Patch size (only 16 supported).
        act: Activation name forwarded to ``ConvNormLayerFusable``.
    """

    def __init__(self, embed_dim: int = 192, patch_size: int = 16, act: str = "relu") -> None:
        super().__init__()
        if patch_size != 16:
            msg = "Only patch_size=16 is supported for ConvPyramidPatchEmbed"
            raise ValueError(msg)

        num_stages = int(math.log2(patch_size)) - 1  # 3
        ratios = [2**i for i in range(num_stages, 0, -1)]  # [8, 4, 2]
        channels = [embed_dim // r for r in ratios]  # [24, 48, 96] for embed_dim=192

        in_ch_list = [3] + channels[:-1]
        self.convs = nn.ModuleList(
            [
                ConvNormLayerFusable(in_ch, out_ch, 3, 2, act=_build_act(act))
                for in_ch, out_ch in zip(in_ch_list, channels)
            ]
        )
        self.proj = nn.Conv2d(channels[-1], embed_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for conv in self.convs:
            x = conv(x)
        return self.proj(x)


def _build_act(act: str) -> type[nn.Module] | None:
    """Map activation string to nn.Module class."""
    _act_map: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    return _act_map.get(act.lower())


# ---------------------------------------------------------------------------
# ViT components
# ---------------------------------------------------------------------------


class Mlp(nn.Module):
    """MLP block (SiLU activation by default)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.SiLU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.fc2(self.drop(self.act(self.fc1(x))))


class VisionTransformer(nn.Module):
    """ViT backbone used by EC-ViT models.

    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        ffn_ratio: MLP hidden-dim ratio.
        qkv_bias: Whether to use bias in QKV projection.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth drop path rate.
        patch_size: Patch size (only 16 supported).
        return_layers: Block indices whose output tokens are collected.
        norm_layer: Norm layer class.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        patch_size: int = 16,
        return_layers: list[int] | None = None,
        norm_layer: type[nn.Module] | Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if return_layers is None:
            return_layers = [10, 11]
        self.embed_dim = embed_dim
        self.return_layers = return_layers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = ConvPyramidPatchEmbed(embed_dim=embed_dim, patch_size=patch_size)
        self.patch_size = patch_size
        self.register_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,  # type: ignore[arg-type]
                    mlp_layer=Mlp,
                )
                for i in range(depth)
            ]
        )

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100.0,
            normalize_coords="separate",
        )
        self._init_weights()

    def _init_weights(self) -> None:
        self.apply(self._init_vit_weights)
        self.rope_embed._init_weights()  # noqa: SLF001
        trunc_normal_(self.register_token, std=0.02)

    @staticmethod
    def _init_vit_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)  # type: ignore[arg-type]
            nn.init.ones_(m.weight)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass, returns patch tokens for each ``return_layers`` block."""
        x_embed = self.patch_embed(x)
        _, _, H, W = x_embed.shape  # noqa: N806
        x_embed = x_embed.flatten(2).transpose(1, 2)  # [B, N, C]
        reg = self.register_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((reg, x_embed), dim=1)
        rope = self.rope_embed(h=H, w=W)

        outs: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos=rope)
            if i in self.return_layers:
                outs.append(x[:, 1:])  # exclude register token
        return outs


# ---------------------------------------------------------------------------
# ViTAdapter  (= ECViTAdapter)
# ---------------------------------------------------------------------------

_BACKBONE_CONFIGS: dict[str, dict] = {
    # ECDet backbones
    "ecvitt": {"embed_dim": 192, "num_heads": 3, "depth": 12, "ffn_ratio": 4},
    "ecvittplus": {"embed_dim": 256, "num_heads": 4, "depth": 12, "ffn_ratio": 4},
    "ecvits": {"embed_dim": 384, "num_heads": 6, "depth": 12, "ffn_ratio": 4},
    "ecvitsplus": {"embed_dim": 384, "num_heads": 6, "depth": 12, "ffn_ratio": 6},
    # ECSeg backbones
    "ecseg_vitt": {"embed_dim": 192, "num_heads": 3, "depth": 12, "ffn_ratio": 4},
    "ecseg_vittplus": {"embed_dim": 256, "num_heads": 4, "depth": 12, "ffn_ratio": 4},
    "ecseg_vits": {"embed_dim": 384, "num_heads": 6, "depth": 12, "ffn_ratio": 4},
    "ecseg_vitsplus": {"embed_dim": 384, "num_heads": 6, "depth": 12, "ffn_ratio": 6},
}


class ECViTAdapter(nn.Module):
    """ViT-based backbone for EdgeCrafter detection/segmentation models.

    Wraps a ``VisionTransformer`` and projects its multi-scale outputs to the
    feature pyramid expected by ``HybridEncoder``.  Three spatial levels are
    produced by interpolating the fused ViT output at scales x2, x1, x0.5
    relative to the ViT feature-map resolution.

    Args:
        model_name: One of the keys in ``_BACKBONE_CONFIGS`` (e.g. ``"ecvitt"``).
        interaction_indexes: Indices of ViT blocks whose outputs are fused.
        proj_dim: If not None, all levels are projected from ``embed_dim`` to
            ``proj_dim`` (used for L/X variants with larger backbones).
        num_levels: Number of output pyramid levels (always 3).
        patch_size: Patch size of the ViT (only 16 supported).
    """

    _pretrained_urls: ClassVar[dict[str, str]] = {
        "ecvitt": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitt.pth",
        "ecvittplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvittplus.pth",
        "ecvits": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvits.pth",
        "ecvitsplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitsplus.pth",
        "ecseg_vitt": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vitt.pth",
        "ecseg_vittplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vittplus.pth",
        "ecseg_vits": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vits.pth",
        "ecseg_vitsplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vitsplus.pth",
    }

    def __init__(
        self,
        model_name: str,
        interaction_indexes: list[int] | None = None,
        proj_dim: int | None = None,
        num_levels: int = 3,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        if model_name not in _BACKBONE_CONFIGS:
            msg = f"Unknown ECViT model name '{model_name}'. Available: {list(_BACKBONE_CONFIGS)}"
            raise ValueError(msg)

        cfg = _BACKBONE_CONFIGS[model_name]
        embed_dim: int = cfg["embed_dim"]

        if interaction_indexes is None:
            interaction_indexes = [10, 11]

        self.backbone = VisionTransformer(
            embed_dim=embed_dim,
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            ffn_ratio=cfg["ffn_ratio"],
            return_layers=interaction_indexes,
            patch_size=patch_size,
        )

        if num_levels != 3:
            msg = "Only num_levels=3 is supported for ECViTAdapter"
            raise ValueError(msg)

        self.num_levels = num_levels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # When proj_dim is set: project all levels from embed_dim → proj_dim.
        # When None: a single projector reduces only the coarsest level, keeping
        # all levels at embed_dim (matching the S/M checkpoint key layout).
        if proj_dim is not None:
            self.projector = nn.ModuleList([ConvNormLayerFusable(embed_dim, proj_dim, 1, 1) for _ in range(num_levels)])
            self.out_channels = proj_dim
        else:
            self.projector = nn.ModuleList([ConvNormLayerFusable(embed_dim, embed_dim, 1, 1)])
            self.out_channels = embed_dim

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return a 3-level feature pyramid from the ViT backbone."""
        H_c = x.shape[2] // self.patch_size  # noqa: N806
        W_c = x.shape[3] // self.patch_size  # noqa: N806
        bs = x.shape[0]

        return_layers = self.backbone(x)
        fused = torch.mean(torch.stack(return_layers), dim=0)  # [B, N, C]
        fused = fused.transpose(1, 2).contiguous().view(bs, -1, H_c, W_c)  # [B, C, H, W]

        proj_feats: list[torch.Tensor] = []
        for i in range(self.num_levels):
            scale = 2 ** (1 - i)
            feat = F.interpolate(
                fused,
                size=[int(H_c * scale), int(W_c * scale)],
                mode="bilinear",
                align_corners=False,
            )
            proj_feats.append(feat)

        # Project only the last (coarsest) level when using a single projector
        if len(self.projector) == 1:
            proj_feats[-1] = self.projector[0](proj_feats[-1])
        else:
            proj_feats = [layer(feat) for layer, feat in zip(self.projector, proj_feats)]

        return proj_feats
