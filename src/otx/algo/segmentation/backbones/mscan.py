# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MSCAN backbone for SegNext model."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
from torch import nn
from torch.nn import SyncBatchNorm

from otx.algo.modules import build_norm_layer
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
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initializes the MLP module."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = activation()
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
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(build_norm_layer, SyncBatchNorm, requires_grad=True)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Callable[..., nn.Module] = nn.GELU,
        normalization: Callable[..., nn.Module] = partial(build_norm_layer, SyncBatchNorm, requires_grad=True),
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(normalization, num_features=out_channels // 2)[1],
            activation(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(normalization, num_features=out_channels)[1],
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
    """Spatial Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        in_channels (int): The number of input channels.
        attention_kernel_sizes (List[Union[int, List[int]]]): The size of attention kernels.
        attention_kernel_paddings (List[Union[int, List[int]]]): The paddings of attention kernels.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
    """

    def __init__(
        self,
        in_channels: int,
        attention_kernel_sizes: list[int | list[int]] = [5, [1, 7], [1, 11], [1, 21]],  # noqa: B006
        attention_kernel_paddings: list[int | list[int]] = [2, [0, 3], [0, 5], [0, 10]],  # noqa: B006
        activation: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        """Init the MSCASpatialAttention module."""
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)  # type: nn.Conv2d
        self.activation = activation()  # type: nn.Module
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

    Args:
        channels (int): The number of input channels.
        attention_kernel_sizes (List[Union[int, List[int]]]): The size of attention kernels.
        attention_kernel_paddings (List[Union[int, List[int]]]): The paddings of attention kernels.
        mlp_ratio (float): The ratio of the number of hidden units in the MLP to the number of input channels.
        drop (float): The dropout rate.
        drop_path (float): The dropout rate for the path.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(build_norm_layer, SyncBatchNorm, requires_grad=True)``.
    """

    def __init__(
        self,
        channels: int,
        attention_kernel_sizes: list[int | list[int]] = [5, [1, 7], [1, 11], [1, 21]],  # noqa: B006
        attention_kernel_paddings: list[int | list[int]] = [2, [0, 3], [0, 5], [0, 10]],  # noqa: B006
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        activation: Callable[..., nn.Module] = nn.GELU,
        normalization: Callable[..., nn.Module] = partial(build_norm_layer, SyncBatchNorm, requires_grad=True),
    ) -> None:
        """Initialize a MSCABlock."""
        super().__init__()
        self.norm1 = build_norm_layer(normalization, num_features=channels)[1]  # type: nn.Module
        self.attn = MSCASpatialAttention(
            channels,
            attention_kernel_sizes,
            attention_kernel_paddings,
            activation,
        )  # type: MSCAAttention
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # type: nn.Module
        self.norm2 = build_norm_layer(normalization, num_features=channels)[1]  # type: nn.Module
        mlp_hidden_channels = int(channels * mlp_ratio)  # type: int
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            activation=activation,
            drop=drop,
        )  # type: Mlp
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
    """Image to Patch Embedding.

    Args:
        patch_size (int, optional): The patch size. Defaults to 7.
        stride (int, optional): Stride of the convolutional layer. Defaults to 4.
        in_channels (int, optional): The number of input channels. Defaults to 3.
        embed_dim (int, optional): The dimensions of embedding. Defaults to 768.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(build_norm_layer, SyncBatchNorm, requires_grad=True)``.
    """

    def __init__(
        self,
        patch_size: int = 7,
        stride: int = 4,
        in_channels: int = 3,
        embed_dim: int = 768,
        normalization: Callable[..., nn.Module] = partial(build_norm_layer, SyncBatchNorm, requires_grad=True),
    ):
        """Initializes the OverlapPatchEmbed module."""
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = build_norm_layer(normalization, num_features=embed_dim)[1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Forward function."""
        x = self.proj(x)
        _, _, h, w = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, h, w


class NNMSCAN(nn.Module):
    """SegNeXt Multi-Scale Convolutional Attention Network (MCSAN) backbone.

    This backbone is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

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
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(build_norm_layer, SyncBatchNorm, requires_grad=True)``.
        init_cfg (Optional[Union[Dict[str, str], List[Dict[str, str]]]]): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: list[int] = [64, 128, 320, 512],  # noqa: B006
        mlp_ratios: list[int] = [8, 8, 4, 4],  # noqa: B006
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        depths: list[int] = [3, 4, 6, 3],  # noqa: B006
        num_stages: int = 4,
        attention_kernel_sizes: list[int | list[int]] = [5, [1, 7], [1, 11], [1, 21]],  # noqa: B006
        attention_kernel_paddings: list[int | list[int]] = [2, [0, 3], [0, 5], [0, 10]],  # noqa: B006
        activation: Callable[..., nn.Module] = nn.GELU,
        normalization: Callable[..., nn.Module] = partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True),
        pretrained_weights: str | None = None,
    ) -> None:
        """Initialize a MSCAN backbone."""
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(in_channels, embed_dims[0], normalization=normalization)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    normalization=normalization,
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
                        activation=activation,
                        normalization=normalization,
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
            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            checkpoint = load_from_http(filename=pretrained, map_location="cpu", model_dir=cache_dir)
            print(f"init weight - {pretrained}")
        if checkpoint is not None:
            load_checkpoint_to_model(self, checkpoint, prefix=prefix)


class MSCAN:
    """MSCAN backbone factory."""

    MSCAN_CFG: ClassVar[dict[str, Any]] = {
        "segnext_tiny": {
            "depths": [3, 3, 5, 2],
            "embed_dims": [32, 64, 160, 256],
            "pretrained_weights": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth",
        },
        "segnext_small": {
            "depths": [2, 2, 4, 2],
            "pretrained_weights": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth",
        },
        "segnext_base": {
            "depths": [3, 3, 12, 3],
            "pretrained_weights": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth",
        },
    }

    def __new__(cls, version: str) -> NNMSCAN:
        """Constructor for MSCAN backbone."""
        if version not in cls.MSCAN_CFG:
            msg = f"model type '{version}' is not supported"
            raise KeyError(msg)

        return NNMSCAN(**cls.MSCAN_CFG[version])
