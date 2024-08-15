# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TinyViT model for the OTX visual prompting."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F  # noqa: N812
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor, nn

from otx.algo.visual_prompting.utils.layer_norm_2d import LayerNorm2d


class Conv2d_BN(nn.Sequential):  # noqa: N801
    """Conv2d_BN for TinyViT."""

    def __init__(
        self,
        a: int,
        b: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        """Fuse weights and biases."""
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchEmbed(nn.Module):
    """PatchEmbed for TinyViT."""

    def __init__(self, in_chans: int, embed_dim: int, resolution: int, activation: nn.Module) -> None:
        super().__init__()
        img_size: tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        return self.seq(x)


class MBConv(nn.Module):
    """MBConv for TinyViT."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        expand_ratio: float,
        activation: nn.Module,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        return self.act3(x)


class PatchMerging(nn.Module):
    """PatchMerging for TinyViT."""

    def __init__(self, input_resolution: tuple[int, int], dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if out_dim in (320, 448, 576):
            stride_c = 1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        if x.ndim == 3:
            height, width = self.input_resolution
            batch = len(x)
            x = x.view(batch, height, width, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x.flatten(2).transpose(1, 2)


class ConvLayer(nn.Module):
    """ConvLayer for TinyViT."""

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        depth: int,
        activation: nn.Module,
        drop_path: list[float] | float = 0.0,
        downsample: nn.Module | None = None,
        out_dim: int | None = None,
        conv_expand_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ],
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    """MLP for TinyViT."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    """Attention block for TinyViT."""

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: int = 4,
        resolution: tuple[int, int] = (14, 14),
    ) -> None:
        super().__init__()
        assert isinstance(resolution, tuple)  # noqa: S101
        assert len(resolution) == 2  # noqa: S101
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        n = len(points)
        attention_offsets: dict[tuple[int, int], int] = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(n, n), persistent=False)

    @torch.no_grad()
    def train(self, mode: bool = True) -> None:  # noqa: D102
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.register_buffer("ab", self.attention_biases[:, self.attention_bias_idxs], persistent=False)

    def forward(self, x: Tensor) -> Tensor:  # x (B,N,C)
        """Forward."""
        b, n, _ = x.shape

        # normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (b, n, num_heads, d)
        q, k, v = qkv.view(b, n, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (b, num_heads, n, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, self.dh)
        return self.proj(x)


class TinyViTBlock(nn.Module):
    """TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        activation: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"  # noqa: S101
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"  # noqa: S101
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        h, w = self.input_resolution
        b, l, c = x.shape  # noqa: E741
        if h * w != l:
            msg = f"Input feature has wrong size. Expected that h({h}) * w({w}) == l({l})."
            raise ValueError(msg)
        res_x = x
        if self.window_size == h and self.window_size == w:
            x = self.attn(x)
        else:
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            ph, pw = h + pad_b, w + pad_r
            nh = ph // self.window_size
            hw = pw // self.window_size
            # window partition
            x = (
                x.view(b, nh, self.window_size, hw, self.window_size, c)
                .transpose(2, 3)
                .reshape(b * nh * hw, self.window_size * self.window_size, c)
            )
            x = self.attn(x)
            # window reverse
            x = x.view(b, nh, hw, self.window_size, self.window_size, c).transpose(2, 3).reshape(b, ph, pw, c)

            if padding:
                x = x[:, :h, :w].contiguous()

            x = x.view(b, l, c)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.local_conv(x)
        x = x.view(b, c, l).transpose(1, 2)

        return x + self.drop_path(self.mlp(x))

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: list[float] | float = 0.0,
        downsample: nn.Module | None = None,
        local_conv_size: int = 3,
        activation: nn.Module = nn.GELU,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(depth)
            ],
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:  # noqa: D102
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(nn.Module):
    """TinyViT for MobileSAM."""

    def __init__(
        self,
        img_size: int = 1024,
        in_chans: int = 3,
        embed_dims: list[int] | None = None,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_sizes: list[int] | None = None,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        layer_lr_decay: float = 1.0,
    ) -> None:
        super().__init__()
        embed_dims = embed_dims or [96, 192, 384, 768]
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]
        window_sizes = window_sizes or [7, 7, 14, 7]

        self.img_size = img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            resolution=img_size,
            activation=activation,
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = {
                "dim": embed_dims[i_layer],
                "input_resolution": (
                    patches_resolution[0] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                    patches_resolution[1] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                ),
                "depth": depths[i_layer],
                "drop_path": dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                "downsample": PatchMerging if (i_layer < self.num_layers - 1) else None,
                "out_dim": embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                "activation": activation,
            }
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay: float) -> None:
        """Set layer lr decay."""
        decay_rate = layer_lr_decay

        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m: nn.Module, scale: float) -> None:
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))  # noqa: B023
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))  # noqa: B023
        assert i == depth  # noqa: S101

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m: nn.Module) -> None:
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name  # noqa: S101

        self.apply(_check_lr_scale)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set[str]:
        """Keyworkds for no weight decay."""
        return {"attention_biases"}

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        batch, _, channel = x.size()
        x = x.view(batch, self.img_size // 16, self.img_size // 16, channel)
        x = x.permute(0, 3, 1, 2)
        return self.neck(x)
