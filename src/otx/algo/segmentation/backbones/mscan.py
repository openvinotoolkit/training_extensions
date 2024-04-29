# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MSCAN backbone for SegNext model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from otx.algo.modules import build_activation_layer, build_norm_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.utils.mmengine_utils import load_checkpoint_to_model, load_from_http

if TYPE_CHECKING:
    from torch import Tensor


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample.

    Args:
        x (Tensor): The input tensor.
        drop_prob (float): Probability of the path to be zeroed. Default: 0.0
        training (bool): The running mode. Default: False
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    """DropPath."""

    def __init__(self, drop_prob: float = 0.1):
        """Drop paths (Stochastic Depth) per sample.

        Args:
            drop_prob (float): Probability of the path to be zeroed. Default: 0.1
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return drop_path(x, self.drop_prob, self.training)


class Mlp(BaseModule):
    """Multi Layer Perceptron (MLP) Module.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features.
            Defaults: None.
        out_features (int): The dimension of output features.
            Defaults: None.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_cfg: dict[str, str] | None = None,
        drop: float = 0.0,
    ) -> None:
        """Initializes the MLP module.

        Args:
            in_features (int): The dimension of the input features.
            hidden_features (Optional[int]): The dimension of the hidden features.
                Defaults to None.
            out_features (Optional[int]): The dimension of the output features.
                Defaults to None.
            act_cfg (Dict[str, str] | None): Config dict for the activation layer in the block.
                Defaults to {"type": "GELU"} if None.
            drop (float): The dropout rate in the MLP block.
                Defaults to 0.0.
        """
        super().__init__()
        if act_cfg is None:
            act_cfg = {"type": "GELU"}
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        return self.drop(x)


class StemConv(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): The dimension of input channels.
        out_channels (int): The dimension of output channels.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_cfg: dict[str, str] | None = None,
        norm_cfg: dict[str, str | bool] | None = None,
    ) -> None:
        """Stem Block at the beginning of Semantic Branch.

        Args:
            in_channels (int): The dimension of input channels.
            out_channels (int): The dimension of output channels.
            act_cfg (Dict[str, str] | None): Config dict for activation layer in block.
                Default: dict(type='GELU') if None.
            norm_cfg (Dict[str, Union[str, bool]] | None): Config dict for normalization layer.
                Defaults: dict(type='SyncBN', requires_grad=True) if None.
        """
        super().__init__()
        if act_cfg is None:
            act_cfg = {"type": "GELU"}
        if norm_cfg is None:
            norm_cfg = {"type": "SyncBN", "requires_grad": True}

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Forward function."""
        x = self.proj(x)
        _, _, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


class MSCAAttention(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA)."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: list[Any] = [5, [1, 7], [1, 11], [1, 21]],  # noqa: B006
        paddings: list[Any] = [2, [0, 3], [0, 5], [0, 10]],  # noqa: B006
    ) -> None:
        """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

        Args:
            channels (int): The dimension of channels.
            kernel_sizes (List[Union[int, List[int]]]): The size of attention kernel.
                Defaults: [5, [1, 7], [1, 11], [1, 21]].
            paddings (List[Union[int, List[int]]]): The number of
                corresponding padding value in attention module.
                Defaults: [2, [0, 3], [0, 5], [0, 10]].
        """
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=kernel_sizes[0], padding=paddings[0], groups=channels)
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f"conv{i}_1", f"conv{i}_2"]
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_, conv_name):
                self.add_module(i_conv, nn.Conv2d(channels, channels, tuple(i_kernel), padding=i_pad, groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention

        return attn * u


class MSCASpatialAttention(BaseModule):
    """Spatial Attention Module in Multi-Scale Convolutional Attention Module (MSCA)."""

    def __init__(
        self,
        in_channels: int,
        attention_kernel_sizes: list[int | list[int]] = [5, [1, 7], [1, 11], [1, 21]],  # noqa: B006
        attention_kernel_paddings: list[int | list[int]] = [2, [0, 3], [0, 5], [0, 10]],  # noqa: B006
        act_cfg: dict[str, str] | None = None,
    ) -> None:
        """Init the MSCASpatialAttention module.

        Args:
            in_channels (int): The number of input channels.
            attention_kernel_sizes (List[Union[int, List[int]]]): The size of attention kernels.
            attention_kernel_paddings (List[Union[int, List[int]]]): The paddings of attention kernels.
            act_cfg (Dict[str, str] | None): The config of activation layer.
        """
        super().__init__()
        if act_cfg is None:
            act_cfg = {"type": "GELU"}
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)  # type: nn.Conv2d
        self.activation = build_activation_layer(act_cfg)  # type: nn.Module
        self.spatial_gating_unit = MSCAAttention(in_channels, attention_kernel_sizes, attention_kernel_paddings)  # type: MSCAAttention
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)  # type: nn.Conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut


class MSCABlock(BaseModule):
    """Basic Multi-Scale Convolutional Attention Block.

    It leverage the large kernel attention (LKA) mechanism to build both channel and spatial
    attention. In each branch, it uses two depth-wise strip convolutions to
    approximate standard depth-wise convolutions with large kernels. The kernel
    size for each branch is set to 7, 11, and 21, respectively.
    """

    def __init__(
        self,
        channels: int,
        attention_kernel_sizes: list[int | list[int]] = [5, [1, 7], [1, 11], [1, 21]],  # noqa: B006
        attention_kernel_paddings: list[int | list[int]] = [2, [0, 3], [0, 5], [0, 10]],  # noqa: B006
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_cfg: dict[str, str] | None = None,
        norm_cfg: dict[str, str | bool] | None = None,
    ) -> None:
        """Initialize a MSCABlock.

        Args:
            channels (int): The number of input channels.
            attention_kernel_sizes (List[Union[int, List[int]]]): The size of attention kernels.
            attention_kernel_paddings (List[Union[int, List[int]]]): The paddings of attention kernels.
            mlp_ratio (float): The ratio of the number of hidden units in the MLP to the number of input channels.
            drop (float): The dropout rate.
            drop_path (float): The dropout rate for the path.
            act_cfg (Dict[str, str] | None): The config of activation layer.
            norm_cfg (Dict[str, Union[str, bool]] | None): The config of normalization layer.
        """
        super().__init__()
        if act_cfg is None:
            act_cfg = {"type": "GELU"}
        if norm_cfg is None:
            norm_cfg = {"type": "SyncBN", "requires_grad": True}
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]  # type: nn.Module
        self.attn = MSCASpatialAttention(channels, attention_kernel_sizes, attention_kernel_paddings, act_cfg)  # type: MSCAAttention
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # type: nn.Module
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]  # type: nn.Module
        mlp_hidden_channels = int(channels * mlp_ratio)  # type: int
        self.mlp = Mlp(in_features=channels, hidden_features=mlp_hidden_channels, act_cfg=act_cfg, drop=drop)  # type: Mlp
        layer_scale_init_value = 1e-2  # type: float
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)  # type: nn.Parameter
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)  # type: nn.Parameter

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Forward function."""
        b, n, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, h, w)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x.view(b, c, n).permute(0, 2, 1)


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding."""

    def __init__(
        self,
        patch_size: int = 7,
        stride: int = 4,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_cfg: dict[str, Any] | None = None,
    ):
        """Initializes the OverlapPatchEmbed module.

        Args:
            patch_size (int, optional): The patch size. Defaults to 7.
            stride (int, optional): Stride of the convolutional layer. Defaults to 4.
            in_channels (int, optional): The number of input channels. Defaults to 3.
            embed_dim (int, optional): The dimensions of embedding. Defaults to 768.
            norm_cfg (dict[str, Any] | None, optional): Config dict for normalization layer.
                Defaults to None. If None, {"type": "SyncBN", "requires_grad": True} is used.
        """
        super().__init__()
        if norm_cfg is None:
            norm_cfg = {"type": "SyncBN", "requires_grad": True}

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Forward function."""
        x = self.proj(x)
        _, _, h, w = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, h, w


class MSCAN(BaseModule):
    """SegNeXt Multi-Scale Convolutional Attention Network (MCSAN) backbone.

    This backbone is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: list[int] = [64, 128, 256, 512],  # noqa: B006
        mlp_ratios: list[int] = [4, 4, 4, 4],  # noqa: B006
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        depths: list[int] = [3, 4, 6, 3],  # noqa: B006
        num_stages: int = 4,
        attention_kernel_sizes: list[int | list[int]] = [5, [1, 7], [1, 11], [1, 21]],  # noqa: B006
        attention_kernel_paddings: list[int | list[int]] = [2, [0, 3], [0, 5], [0, 10]],  # noqa: B006
        act_cfg: dict[str, str] | None = None,
        norm_cfg: dict[str, str | bool] | None = None,
        init_cfg: dict[str, str] | list[dict[str, str]] | None = None,
        pretrained_weights: str | None = None,
    ) -> None:
        """Initialize a MSCAN backbone.

        Args:
            in_channels (int): The number of input channels. Defaults to 3.
            embed_dims (List[int]): Embedding dimension. Defaults to [64, 128, 256, 512].
            mlp_ratios (List[int]): Ratio of mlp hidden dim to embedding dim. Defaults to [4, 4, 4, 4].
            drop_rate (float): Dropout rate. Defaults to 0.0.
            drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
            depths (List[int]): Depths of each Swin Transformer stage. Defaults to [3, 4, 6, 3].
            num_stages (int): MSCAN stages. Defaults to 4.
            attention_kernel_sizes (List[Union[int, List[int]]]): Size of attention kernel in
                Attention Module (Figure 2(b) of original paper). Defaults to [5, [1, 7], [1, 11], [1, 21]].
            attention_kernel_paddings (List[Union[int, List[int]]]): Size of attention paddings
                in Attention Module (Figure 2(b) of original paper). Defaults to [2, [0, 3], [0, 5], [0, 10]].
            act_cfg (Dict[str, str] | None): Config dict for activation layer in block.
                Defaults to dict(type='GELU') if None.
            norm_cfg (Dict[str, Union[str, bool]] | None): Config dict for normalization layer.
                Defaults to dict(type='SyncBN', requires_grad=True) if None.
            init_cfg (Optional[Union[Dict[str, str], List[Dict[str, str]]]]): Initialization config dict.
                Defaults to None.
        """
        super().__init__(init_cfg=init_cfg)
        if act_cfg is None:
            act_cfg = {"type": "GELU"}
        if norm_cfg is None:
            norm_cfg = {"type": "SyncBN", "requires_grad": True}

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(in_channels, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg,
                )
            block = nn.ModuleList(
                [
                    MSCABlock(
                        channels=embed_dims[i],
                        attention_kernel_sizes=attention_kernel_sizes,
                        attention_kernel_paddings=attention_kernel_paddings,
                        mlp_ratio=mlp_ratios[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                    )
                    for j in range(depths[i])
                ],
            )
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward function."""
        b = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, h, w = patch_embed(x)
            for blk in block:
                x = blk(x, h, w)
            x = norm(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def load_pretrained_weights(self, pretrained: str | None = None, prefix: str = "") -> None:
        """Initialize weights."""
        checkpoint = None
        if isinstance(pretrained, str) and Path(pretrained).exists():
            checkpoint = torch.load(pretrained, "cpu")
            print(f"init weight - {pretrained}")
        elif pretrained is not None:
            checkpoint = load_from_http(pretrained, "cpu")
            print(f"init weight - {pretrained}")
        if checkpoint is not None:
            load_checkpoint_to_model(self, checkpoint, prefix=prefix)
