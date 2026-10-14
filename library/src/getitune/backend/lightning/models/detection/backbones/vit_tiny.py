# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Vision Transformer (ViT) Tiny implementation for object detection.

Modified from DEIMv2 (https://github.com/Intellindust-AI-Lab/DEIMv2)
Modified from https://huggingface.co/spaces/Hila/RobustViT/blob/main/ViT/ViT_new.py

This implementation is licensed under Apache-2.0. Pretrained ViT-Tiny weights
are distilled from DINOv3 and are subject to the DINOv3 license.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn

from getitune.backend.lightning.models.common.layers.position_embed import RopePositionEmbedding
from getitune.backend.lightning.models.common.layers.vit_blocks import Attention, Block  # noqa: F401
from getitune.backend.lightning.models.utils.weight_init import trunc_normal_


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input for RoPE.

    Splits the last dimension in half and swaps the two halves with negation.

    Args:
        x: Input tensor of shape (..., D) where D is even.

    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Apply Rotary Position Embedding (RoPE) to the input tensor.

    Args:
        x: Input tensor to apply RoPE to.
        sin: Precomputed sine values for position encoding.
        cos: Precomputed cosine values for position encoding.

    Returns:
        Tensor with RoPE applied.
    """
    return (x * cos) + (rotate_half(x) * sin)


class SimplifiedPatchEmbed(nn.Module):
    """Patch Embedding layer for Vision Transformer.

    Converts an image into a sequence of patch embeddings using a convolutional layer.

    Args:
        img_size: Input image size. Defaults to 224.
        patch_size: Size of each patch. Defaults to 16.
        in_chans: Number of input channels. Defaults to 3.
        embed_dim: Embedding dimension. Defaults to 768.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """Convert image to patch embeddings.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        return self.proj(x).flatten(2).transpose(1, 2)


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Drop paths (Stochastic Depth) per sample.

    When applied in main path of residual blocks, this implements stochastic depth.

    Args:
        x: Input tensor.
        drop_prob: Probability of dropping the path. Defaults to 0.0.
        training: Whether the model is in training mode. Defaults to False.

    Returns:
        Output tensor with drop path applied during training.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    A module wrapper for the drop_path function.

    Args:
        drop_prob: Probability of dropping the path. Defaults to None (0.0).
    """

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Apply drop path to input.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with drop path applied.
        """
        return drop_path(x, self.drop_prob or 0.0, self.training)


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) backbone for object detection.

    A Vision Transformer with Rotary Position Embedding (RoPE) support,
    designed for multi-scale feature extraction.

    Args:
        img_size: Input image size. Defaults to 224.
        patch_size: Size of each patch. Defaults to 16.
        in_chans: Number of input channels. Defaults to 3.
        embed_dim: Embedding dimension. Defaults to 192.
        depth: Number of transformer blocks. Defaults to 12.
        num_heads: Number of attention heads. Defaults to 3.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
        qkv_bias: Whether to add bias to QKV projection. Defaults to True.
        drop_rate: Dropout rate. Defaults to 0.0.
        attn_drop_rate: Attention dropout rate. Defaults to 0.0.
        drop_path_rate: Drop path rate. Defaults to 0.0.
        return_layers: List of layer indices to return features from.
            Defaults to [3, 7, 11].
        embed_layer: Patch embedding layer class. Defaults to SimplifiedPatchEmbed.
        norm_layer: Normalization layer class. Defaults to LayerNorm with eps=1e-6.
        act_layer: Activation layer class. Defaults to nn.GELU.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        return_layers: list[int] | None = None,
        embed_layer: type[nn.Module] = SimplifiedPatchEmbed,
        norm_layer: type[nn.Module] | Callable[..., nn.Module] | None = None,
        act_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if return_layers is None:
            return_layers = [3, 7, 11]
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.return_layers = return_layers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self._model = nn.Module()
        self._model.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.patch_size = patch_size
        self._model.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self._model.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self._model.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100.0,
            normalize_coords="separate",
            shift_coords=None,
            jitter_coords=None,
            rescale_coords=None,
            dtype=None,
            device=None,
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        trunc_normal_(self._model.cls_token, std=0.02)
        self._model.rope_embed._init_weights()  # noqa: SLF001
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m: nn.Module) -> None:
        """Initialize weights for ViT layers.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """Return set of parameter names that should not use weight decay.

        Returns:
            Set of parameter names to exclude from weight decay.
        """
        return {"cls_token"}

    def get_model(self) -> nn.Module:
        """Get the internal model module.

        Returns:
            The internal _model module containing all layers.
        """
        return self._model

    def feature_dim(self) -> int:
        """Get the feature dimension.

        Returns:
            The embedding dimension.
        """
        return self.embed_dim

    def forward(self, x: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Forward pass through Vision Transformer.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            List of tuples (patch_features, cls_token) for each return layer.
            patch_features has shape (B, num_patches, embed_dim).
            cls_token has shape (B, embed_dim).
        """
        outs = []
        B, C, H, W = x.shape  # noqa: N806

        x_embed = self._model.patch_embed(x)
        cls_token = self._model.cls_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((cls_token, x_embed), dim=1)

        patch_grid_h = H // self.patch_size
        patch_grid_w = W // self.patch_size
        rope_sincos = self._model.rope_embed(h=patch_grid_h, w=patch_grid_w)

        for i, blk in enumerate(self._model.blocks):
            x = blk(x, rope_sincos=rope_sincos)
            if i in self.return_layers:
                outs.append((x[:, 1:], x[:, 0]))
        return outs
