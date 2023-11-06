"""TinyViT for MobileSAM."""

# Copyright (c) 2022 Microsoft
# https://github.com/ChaoningZhang/MobileSAM
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools

import torch
from timm.models.layers import DropPath as TimmDropPath
from timm.models.layers import to_2tuple, trunc_normal_
from torch import Tensor, nn
from torch.nn import functional
from torch.utils import checkpoint


class Conv2dBN(nn.Sequential):
    """Conv2dBN for TinyViT."""

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
        """Initializes a TinyViT backbone module with a convolutional layer and batch normalization.

        Args:
            a (int): Number of input channels.
            b (int): Number of output channels.
            ks (int, optional): Kernel size of the convolutional layer. Defaults to 1.
            stride (int, optional): Stride of the convolutional layer. Defaults to 1.
            pad (int, optional): Padding of the convolutional layer. Defaults to 0.
            dilation (int, optional): Dilation of the convolutional layer. Defaults to 1.
            groups (int, optional): Number of groups for the convolutional layer. Defaults to 1.
            bn_weight_init (float, optional): Initial value for the batch normalization weights. Defaults to 1.0.
        """
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


class DropPath(TimmDropPath):
    """DropPath for TinyViT."""

    def __init__(self, drop_prob: list[float] | float | None = None) -> None:
        """Initializes a TinyViT backbone.

        Args:
            drop_prob (list[float] | float | None, optional): The dropout probability for each layer.
                If a list is provided, it should have the same length as the number of layers in the backbone.
                If a float is provided, the same dropout probability will be used for all layers.
                If None, no dropout will be applied. Defaults to None.
        """
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self) -> str:
        """Returns a string representation of the TinyViT model, including the dropout probability.

        Returns:
            str: A string representation of the TinyViT model.
        """
        msg = super().__repr__()
        msg += f"(drop_prob={self.drop_prob})"
        return msg


class PatchEmbed(nn.Module):
    """PatchEmbed for TinyViT."""

    def __init__(self, in_chans: int, embed_dim: int, resolution: int, activation: nn.Module) -> None:
        """Initializes a TinyViT backbone module.

        Args:
            in_chans (int): Number of input channels.
            embed_dim (int): Embedding dimension.
            resolution (int): Resolution of the input image.
            activation (nn.Module): Activation function to use after each convolutional layer.
        """
        super().__init__()
        img_size: tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2dBN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2dBN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward call."""
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
        """Initializes a TinyViT backbone module.

        Args:
            in_chans (int): Number of input channels.
            out_chans (int): Number of output channels.
            expand_ratio (float): Expansion ratio for the hidden channels.
            activation (nn.Module): Activation function to use.
            drop_path (float): Probability of an element to be zeroed in the drop path layer.

        Returns:
            None
        """
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2dBN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2dBN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2dBN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward call."""
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
        """Initializes the TinyViTBackbone module.

        Args:
            input_resolution (tuple[int, int]): The input resolution of the image.
            dim (int): The number of input channels.
            out_dim (int): The number of output channels.
            activation (nn.Module): The activation function to use.
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2dBN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if out_dim in (320, 448, 576):
            stride_c = 1
        self.conv2 = Conv2dBN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2dBN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward call."""
        if x.ndim == 3:
            h, w = self.input_resolution
            b = len(x)
            x = x.view(b, h, w, -1).permute(0, 3, 1, 2)

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
        use_checkpoint: bool = False,
        out_dim: int | None = None,
        conv_expand_ratio: float = 4.0,
    ) -> None:
        """Initializes a Tiny Vision Transformer (ViT) backbone module.

        Args:
            dim (int): the feature dimension.
            input_resolution (int): the input resolution (image size).
            depth (int): the number of blocks in the backbone.
            activation (nn.Module): the activation function to use.
            drop_path (list[float] | float, optional): the drop path probability for each block. Defaults to 0.0.
            downsample (nn.Module | None, optional): the downsample module to use. Defaults to None.
            use_checkpoint (bool, optional): whether to use checkpointing. Defaults to False.
            out_dim (int | None, optional): the output feature dimension. Defaults to None.
            conv_expand_ratio (float, optional): the expansion ratio for the convolutional layers. Defaults to 4.0.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

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
        """Forward call."""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
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
        """Initializes the TinyViTBackbone module.

        Args:
            in_features (int): Number of input features.
            hidden_features (int | None, optional): Number of hidden features. Defaults to None.
            out_features (int | None, optional): Number of output features. Defaults to None.
            act_layer (nn.Module, optional): Activation function. Defaults to nn.GELU.
            drop (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward call."""
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
        """Initializes a TinyViT backbone module.

        Args:
            dim (int): the input feature dimension.
            key_dim (int): the dimension of the query, key, and value vectors.
            num_heads (int, optional): the number of attention heads. Defaults to 8.
            attn_ratio (int, optional): the ratio between the key dimension and the attention dimension. Defaults to 4.
            resolution (tuple[int, int], optional): the spatial resolution of the input feature map.
                Defaults to (14, 14).
        """
        super().__init__()
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
        attention_offsets: dict = {}
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
    def train(self, mode: bool = True) -> None:
        """Sets the module in training mode and removes the attention biases buffer if it exists.

        If not in training mode, the attention biases buffer is created and registered.

        Args:
            mode (bool, optional): Whether to set the module in training mode or not. Default is True.
        """
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.register_buffer("ab", self.attention_biases[:, self.attention_bias_idxs], persistent=False)

    def forward(self, x: Tensor) -> Tensor:  # x (B,N,C)
        """Forward pass of the TinyViT backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dh).
        """
        b, n, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        q, k, v = qkv.view(b, n, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
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
    """TinyViT Block."""

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
        """Initializes a TinyViT backbone module.

        Args:
            dim (int): The feature dimension.
            input_resolution (tuple[int, int]): The input resolution (height, width).
            num_heads (int): The number of attention heads.
            window_size (int, optional): The size of the attention window. Defaults to 7.
            mlp_ratio (float, optional): The ratio of MLP hidden dimension to feature dimension. Defaults to 4.0.
            drop (float, optional): The dropout rate. Defaults to 0.0.
            drop_path (float, optional): The drop path rate. Defaults to 0.0.
            local_conv_size (int, optional): The kernel size of the local convolution. Defaults to 3.
            activation (nn.Module, optional): The activation function. Defaults to nn.GELU.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2dBN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the TinyViTBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        h, w = self.input_resolution
        b, _l, c = x.shape
        res_x = x
        if self.window_size == h and self.window_size == w:
            x = self.attn(x)
        else:
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            p_h, p_w = h + pad_b, w + pad_r
            n_h = p_h // self.window_size
            n_w = p_w // self.window_size
            # window partition
            x = (
                x.view(b, n_h, self.window_size, n_w, self.window_size, c)
                .transpose(2, 3)
                .reshape(b * n_h * n_w, self.window_size * self.window_size, c)
            )
            x = self.attn(x)
            # window reverse
            x = x.view(b, n_h, n_w, self.window_size, self.window_size, c).transpose(2, 3).reshape(b, p_h, p_w, c)

            if padding:
                x = x[:, :h, :w].contiguous()

            x = x.view(b, _l, c)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.local_conv(x)
        x = x.view(b, c, _l).transpose(1, 2)

        return x + self.drop_path(self.mlp(x))

    def extra_repr(self) -> str:
        """Return a string containing the dimensions and parameters of the TinyViT model.

        Returns:
            str: A string containing the dimensions and parameters of the TinyViT model.
        """
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """A basic TinyViT layer for one stage."""

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
        use_checkpoint: bool = False,
        local_conv_size: int = 3,
        activation: nn.Module = nn.GELU,
        out_dim: int | None = None,
    ) -> None:
        """Initializes a TinyViTBackbone module.

        Args:
            dim (int): Dimensionality of the token embeddings.
            input_resolution (tuple[int, int]): Spatial resolution of the input image.
            depth (int): Number of blocks in the backbone.
            num_heads (int): Number of attention heads in each block.
            window_size (int): Size of the attention window in each block.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension in each block.
                Defaults to 4.0.
            drop (float, optional): Dropout rate. Defaults to 0.0.
            drop_path (list[float] | float, optional): Dropout rate for each block. If a single value is provided,
                it will be used for all blocks. Defaults to 0.0.
            downsample (nn.Module | None, optional): Downsample module to reduce the spatial resolution of the input.
                Defaults to None.
            use_checkpoint (bool, optional): Whether to use checkpointing to reduce memory usage. Defaults to False.
            local_conv_size (int, optional): Size of the local convolution kernel in each block. Defaults to 3.
            activation (nn.Module, optional): Activation function to use. Defaults to nn.GELU.
            out_dim (int | None, optional): Output dimensionality of the backbone.
                If None, the output dimensionality will be the same as the token embedding dimensionality.
                Defaults to None.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

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
        """Forward call."""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra configuration parameters.

        Returns:
            str: A string representation of the module's extra configuration parameters.
        """
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class LayerNorm2d(nn.Module):
    """2D-Layer Normalize for TinyViT."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Initializes the TinyViT backbone module.

        Args:
            num_channels (int): The number of channels in the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Forward call."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class TinyViT(nn.Module):
    """TinyViT for MobileSAM."""

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: list[int] | None = None,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_sizes: list[int] | None = None,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        layer_lr_decay: float = 1.0,
    ) -> None:
        """Initializes a Tiny Vision Transformer (TinyViT) model.

        Args:
            img_size (int): The size of the input image.
            in_chans (int): The number of input channels.
            num_classes (int): The number of output classes.
            embed_dims (list[int] | None): The embedding dimensions for each layer. If None, default values are used.
            depths (list[int] | None): The number of layers for each stage. If None, default values are used.
            num_heads (list[int] | None): The number of attention heads for each layer. If None, default values are used
            window_sizes (list[int] | None): The window sizes for each layer. If None, default values are used.
            mlp_ratio (float): The ratio of the hidden size of the feedforward network to the embedding size.
            drop_rate (float): The dropout rate.
            drop_path_rate (float): The drop path rate.
            use_checkpoint (bool): Whether to use checkpointing to save memory.
            mbconv_expand_ratio (float): The expansion ratio for the MobileNetV3 convolutional layers.
            local_conv_size (int): The kernel size for the local convolutional layers.
            layer_lr_decay (float): The learning rate decay factor for each layer.
        """
        super().__init__()
        embed_dims = [96, 192, 384, 768] if embed_dims is None else embed_dims
        depths = [2, 2, 6, 2] if depths is None else depths
        num_heads = [3, 6, 12, 24] if num_heads is None else num_heads
        window_sizes = [7, 7, 14, 7] if window_sizes is None else window_sizes
        self.img_size = img_size
        self.num_classes = num_classes
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
                #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                #                     patches_resolution[1] // (2 ** i_layer)),
                "depth": depths[i_layer],
                "drop_path": dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                "downsample": PatchMerging if (i_layer < self.num_layers - 1) else None,
                "use_checkpoint": use_checkpoint,
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

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m: nn.Module, scale: float) -> None:
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x, scale=lr_scales[i]: _set_lr_scale(x, scale))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x, scale=lr_scales[i - 1]: _set_lr_scale(x, scale))

        for k, p in self.named_parameters():
            p.param_name = k

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

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward call.

        Args:
            x (Tensor): Input image tensor with shape (B, C, H, W).

        Returns:
            Tensor: Output tensor with shape (B, H', W', C').
        """
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        b, _, c = x.size()
        x = x.view(b, 64, 64, c)
        x = x.permute(0, 3, 1, 2)
        return self.neck(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward call.

        Args:
            x (Tensor): Input image tensor with shape (B, C, H, W).

        Returns:
            Tensor: Output tensor with shape (B, H', W', C').
        """
        return self.forward_features(x)


def build_tiny_vit(img_size: int = 1024, drop_path_rate: float = 0.0) -> TinyViT:
    """Build TinyViT backbone.

    Args:
        img_size (int): Input image size.
        drop_path_rate (float): Drop path rate for stochastic depth.

    Returns:
        TinyViT: TinyViT backbone.
    """
    return TinyViT(
        img_size=img_size,
        num_classes=1,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )
