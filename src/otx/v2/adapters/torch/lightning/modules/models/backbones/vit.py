"""Vision Transformers."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional

from otx.v2.adapters.torch.lightning.modules.models.utils import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
class ViT(nn.Module):
    """Vision Transformer for visual prompting task."""

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
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        """Initializes a Vision Transformer (ViT) backbone module.

        Args:
            img_size (int): The size of the input image.
            patch_size (int): The size of the patch to be extracted from the input image.
            in_chans (int): The number of input channels.
            embed_dim (int): The dimension of the embedding.
            depth (int): The number of transformer blocks.
            num_heads (int): The number of attention heads.
            mlp_ratio (float): The ratio of the hidden size of the feedforward network to the embedding size.
            out_chans (int): The number of output channels.
            qkv_bias (bool): Whether to include bias terms in the query, key, and value projections.
            norm_layer (nn.Module): The normalization layer to use.
            act_layer (nn.Module): The activation layer to use.
            use_abs_pos (bool): Whether to use absolute positional embeddings.
            use_rel_pos (bool): Whether to use relative positional embeddings.
            rel_pos_zero_init (bool): Whether to initialize the relative positional embeddings to zero.
            window_size (int): The size of the window for relative positional embeddings.
            global_attn_indexes (tuple[int, ...]): The indices of the transformer blocks to use global attention.
        """
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
    """Transformer blocks with support of window attention and residual propagation blocks."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """Initializes Block Module.

        Args:
            dim (int): The number of channels in the input tensor.
            num_heads (int): The number of attention heads to use in the multi-head attention layer.
            mlp_ratio (float, optional): The ratio of the hidden size of the feedforward network to the embedding size.
                Defaults to 4.0.
            qkv_bias (bool, optional): Whether to include bias terms in the query, key, and value projections.
                Defaults to True.
            norm_layer (type[nn.Module], optional): The normalization layer to use. Defaults to nn.LayerNorm.
            act_layer (type[nn.Module], optional): The activation function to use. Defaults to nn.GELU.
            use_rel_pos (bool, optional): Whether to use relative positional encoding. Defaults to False.
            rel_pos_zero_init (bool, optional): Whether to initialize the relative positional encoding with zeros.
                Defaults to True.
            window_size (int, optional): The size of the window to use for local attention. If 0, global attention
                is used. Defaults to 0.
            input_size (tuple[int, int] | None, optional): The size of the input tensor. If `window_size` is greater
                than 0, this should be set to `(height, width)` of the input tensor. Defaults to None.
        """
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
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (h, w))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """Initializes a Vision Transformer (ViT) backbone module.

        Args:
            dim (int): the feature dimension of the input tensor.
            num_heads (int, optional): the number of attention heads. Defaults to 8.
            qkv_bias (bool, optional): whether to include bias terms in the QKV linear layers. Defaults to True.
            use_rel_pos (bool, optional): whether to use relative positional embeddings. Defaults to False.
            rel_pos_zero_init (bool, optional): whether to initialize the relative positional embeddings to zero.
                Defaults to True.
            input_size (tuple[int, int] | None, optional): the input image size (height, width).
                If use_rel_pos is True, this argument is required. Defaults to None.
        """
        _ = rel_pos_zero_init
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos and input_size is not None:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape (B, H, W, C).

        Returns:
            Tensor: Output tensor of shape (B, H, W, C).
        """
        b, h, w, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, b * self.num_heads, h * w, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (h, w), (h, w))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(b, self.num_heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, -1)
        return self.proj(x)


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.

    Args:
        x (Tensor): Input tokens with [B, H, W, C].
        window_size (int): Window size.

    Returns:
        windows (Tensor): windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp) (Tuple[int, int]): padded height and width before partition
    """
    b, h, w, c = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    h_p, w_p = h + pad_h, w + pad_w

    x = x.view(b, h_p // window_size, window_size, w_p // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, (h_p, w_p)


def window_unpartition(windows: Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]) -> Tensor:
    """Window unpartition into original sequences and removing padding.

    Args:
        windows (Tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x (Tensor): unpartitioned sequences with [B, H, W, C].
    """
    h_p, w_p = pad_hw
    h, w = hw
    b = windows.shape[0] // (h_p * w_p // window_size // window_size)
    x = windows.view(b, h_p // window_size, w_p // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h_p, w_p, -1)

    if h_p > h or w_p > w:
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
        rel_pos_resized = functional.interpolate(
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
    """Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.

    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    r_h = get_rel_pos(q_h, k_h, rel_pos_h)
    r_w = get_rel_pos(q_w, k_w, rel_pos_w)

    b, _, dim = q.shape
    r_q = q.reshape(b, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, r_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, r_w)

    return (attn.view(b, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        b,
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
        """Initializes a Vision Transformer (ViT) backbone module.

        Args:
            kernel_size (tuple[int, int]): Size of the convolutional kernel. Default is (16, 16).
            stride (tuple[int, int]): Stride of the convolution. Default is (16, 16).
            padding (tuple[int, int]): Padding of the convolution. Default is (0, 0).
            in_chans (int): Number of input channels. Default is 3.
            embed_dim (int): Dimensionality of the output embeddings. Default is 768.
        """
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


def build_vit(backbone: str, image_size: int) -> ViT:
    """Build ViT backbone.

    Args:
        backbone (str): backbone name.
        image_size (int): input image size.

    Returns:
        ViT: ViT backbone.
    """
    model_params: dict[str, dict] = {
        "vit_h": {
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "global_attn_indexes": [7, 15, 23, 31],
        },
        "vit_l": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "global_attn_indexes": [5, 11, 17, 23],
        },
        "vit_b": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "global_attn_indexes": [2, 5, 8, 11],
        },
    }

    return ViT(
        img_size=image_size,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        out_chans=256,
        patch_size=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_rel_pos=True,
        window_size=14,
        embed_dim=model_params[backbone]["embed_dim"],
        depth=model_params[backbone]["depth"],
        num_heads=model_params[backbone]["num_heads"],
        global_attn_indexes=model_params[backbone]["global_attn_indexes"],
    )
