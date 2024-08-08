# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ViT models for the OTX visual prompting."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from otx.algo.visual_prompting.utils import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
class ViT(nn.Module):
    """Vision Transformer for visual prompting task.

    Args:
        img_size (int): Input image size.
        patch_size (int): Patch size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Patch embedding dimension.
        depth (int): Depth of ViT.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        out_chans (int): Number of output channels.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        norm_layer (Callable[..., nn.Module]): Normalization layer.
        act_layer (nn.Module): Activation layer.
        use_abs_pos (bool): If True, use absolute positional embeddings.
        use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
        rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
        window_size (int): Window size for window attention blocks.
        global_attn_indexes (tuple[int, ...]): Indexes for blocks using global attention.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        act_layer: nn.Module = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: nn.Parameter | None = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, out_chans, H, W).
        """
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        return self.neck(x.permute(0, 3, 1, 2))


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        norm_layer (Callable[..., nn.Module] | nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer.
        use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
        rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
        window_size (int): Window size for window attention blocks. If it equals 0, then
            use global attention.
        input_size (tuple(int, int) or None): Input resolution for calculating the relative
            positional parameter size.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Callable[..., nn.Module] | type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape (B, H, W, C).

        Returns:
            Tensor: Output tensor of shape (B, H, W, C).
        """
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            height, width = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (height, width))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
        rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
        input_size (tuple(int, int) or None): Input resolution for calculating the relative
            positional parameter size.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."  # noqa: S101
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape (batch, height, width, C).

        Returns:
            Tensor: Output tensor of shape (batch, height, width, C).
        """
        batch, height, width, _ = x.shape
        # qkv with shape (3, batch, nHead, height * width, C)
        qkv = self.qkv(x).reshape(batch, height * width, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (batch * nHead, height * width, C)
        q, k, v = qkv.reshape(3, batch * self.num_heads, height * width, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width))

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(batch, self.num_heads, height, width, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch, height, width, -1)
        )
        return self.proj(x)


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.

    Args:
        x (Tensor): Input tokens with [batch, height, width, channel].
        window_size (int): Window size.

    Returns:
        windows (Tensor): windows after partition with [batch * num_windows, window_size, window_size, channel].
        (hp, wp) (Tuple[int, int]): padded height and width before partition
    """
    batch, height, width, channel = x.shape

    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = height + pad_h, width + pad_w

    x = x.view(batch, hp // window_size, window_size, wp // window_size, window_size, channel)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channel)
    return windows, (hp, wp)


def window_unpartition(windows: Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]) -> Tensor:
    """Window unpartition into original sequences and removing padding.

    Args:
        windows (Tensor): input tokens with [batch * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (hp, wp).
        hw (Tuple): original height and width (h, w) before padding.

    Returns:
        x (Tensor): unpartitioned sequences with [B, H, W, C].
    """
    hp, wp = pad_hw
    h, w = hw
    batch = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(batch, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, hp, wp, -1)

    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    """Get relative positional embeddings according to the relative positions of query and key sizes.

    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Tensor: Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: Tensor,
    q: Tensor,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> Tensor:
    """Calculate decomposed Relative Positional Embeddings from `mvitv2`.

    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (batch, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = get_rel_pos(q_h, k_h, rel_pos_h)
    rw = get_rel_pos(q_w, k_w, rel_pos_w)

    batch, _, dim = q.shape
    r_q = q.reshape(batch, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)

    return (attn.view(batch, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        batch,
        q_h * q_w,
        k_h * k_w,
    )


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        kernel_size (Tuple): kernel size of the projection layer.
        stride (Tuple): stride of the projection layer.
        padding (Tuple): padding size of the projection layer.
        in_chans (int): Number of input image channels.
        embed_dim (int): Patch embedding dimension.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        """Forward call.

        Args:
            x (Tensor): input image tensor with shape (B, C, H, W).

        Returns:
            Tensor: output tensor with shape (B, H', W', C').
        """
        x = self.proj(x)
        # B C H W -> B H W C
        return x.permute(0, 2, 3, 1)
