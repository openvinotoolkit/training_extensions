# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.backbones.swin.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/backbones/swin.py
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath, to_2tuple
from torch import Tensor, nn

from otx.algo.instance_segmentation.layers import PatchEmbed, PatchMerging
from otx.algo.modules.base_module import BaseModule, ModuleList
from otx.algo.modules.norm import build_norm_layer
from otx.algo.modules.transformer import FFN
from otx.algo.utils.mmengine_utils import load_from_http
from otx.algo.utils.weight_init import constant_init, trunc_normal_, trunc_normal_init

# ruff: noqa: PLR0913


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: tuple[int, int],
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        init_cfg: None = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
        )  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        wh, ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * ww - 1, wh, 1, ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self) -> None:
        """Initialize the weights."""
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Swin Transformer layer computation.

        Args:
            x (Tensor): input features with shape of (num_windows*B, N, C)
            mask (Tensor | None, Optional): mask with shape of (num_windows, Wh*Ww, Wh*Ww), value between (-inf, 0].
        """
        batch_size, num_pred, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_pred, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(batch_size // nw, nw, self.num_heads, num_pred, num_pred) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, num_pred, num_pred)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_pred, channels)
        x = self.proj(x)
        return self.proj_drop(x)

    @staticmethod
    def double_step_seq(step1: int, len1: int, step2: int, len2: int) -> Tensor:
        """Generate double step sequence."""
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop_rate: float = 0,
        proj_drop_rate: float = 0,
        dropout_layer: dict | None = None,
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        if self.shift_size < 0 or self.shift_size >= self.window_size:
            msg = "shift_size must be in [0, window_size)"
            raise ValueError(msg)

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None,
        )

        dropout_layer = {"type": "DropPath", "drop_prob": 0.0} if dropout_layer is None else dropout_layer
        _dropout_layer = deepcopy(dropout_layer)
        dropout_type = _dropout_layer.pop("type")
        if dropout_type != "DropPath":
            msg = "Only support `DropPath` dropout layer."
            raise ValueError(msg)
        self.drop = DropPath(**_dropout_layer)

    def forward(self, query: Tensor, hw_shape: tuple[int, int]) -> Tensor:
        """Forward function."""
        b, length, c = query.shape
        h, w = hw_shape
        if h * w != length:
            msg = "The length of query should be equal to H*W."
            raise ValueError(msg)
        query = query.view(b, h, w, c)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        query = torch.nn.functional.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        h_pad, w_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, h_pad, w_pad, 1), device=query.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for _h in h_slices:
                for _w in w_slices:
                    img_mask[:, _h, _w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, c)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, h_pad, w_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h * w, c)

        return self.drop(x)

    def window_reverse(self, windows: Tensor, h: int, w: int) -> Tensor:
        """Reverse the window partition process.

        Args:
            windows (Tensor): (num_windows*B, window_size, window_size, C)
            h (int): Height of image
            w (int): Width of image
        Returns:
            Tensor: (B, H, W, C)
        """
        window_size = self.window_size
        batch_size = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(batch_size, h // window_size, w // window_size, window_size, window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, h, w, -1)

    def window_partition(self, x: Tensor) -> Tensor:
        """Split x into multi windows.

        Args:
            x (Tensor): (B, H, W, C)

        Returns:
            Tensor: (num_windows*B, window_size, window_size, C)
        """
        batch_size, h, w, c = x.shape
        window_size = self.window_size
        x = x.view(batch_size, h // window_size, window_size, w // window_size, window_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, window_size, window_size, c)


class SwinBlock(BaseModule):
    """Basic Swin Transformer block.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``nn.LayerNorm``.
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        activation: Callable[..., nn.Module] = nn.GELU,
        normalization: Callable[..., nn.Module] = nn.LayerNorm,
        with_cp: bool = False,
        init_cfg: None = None,
    ):
        super().__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(normalization, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer={"type": "DropPath", "drop_prob": drop_path_rate},
            init_cfg=None,
        )

        self.norm2 = build_norm_layer(normalization, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer={"type": "DropPath", "drop_prob": drop_path_rate},
            activation=activation,
            add_identity=True,
            init_cfg=None,
        )

    def forward(self, x: Tensor, hw_shape: Tensor) -> Tensor:
        """Forward function."""

        def _inner_forward(x: Tensor) -> Tensor:
            """Inner forward function."""
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            return self.ffn(x, identity=identity)

        return cp.checkpoint(_inner_forward, x) if self.with_cp and x.requires_grad else _inner_forward(x)


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``nn.LayerNorm``.
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        depth: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: list[float] | float = 0.0,
        downsample: BaseModule | None = None,
        activation: Callable[..., nn.Module] = nn.GELU,
        normalization: Callable[..., nn.Module] = nn.LayerNorm,
        with_cp: bool = False,
        init_cfg: None = None,
    ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            if len(drop_path_rates) != depth:
                msg = "The length of drop_path_rate should be equal to depth."
                raise ValueError(msg)
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=i % 2 != 0,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                activation=activation,
                normalization=normalization,
                with_cp=with_cp,
                init_cfg=None,
            )
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x: Tensor, hw_shape: tuple[int, int]) -> Tensor:
        """Forward function."""
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        return x, hw_shape, x, hw_shape


class SwinTransformer(BaseModule):
    """Swin Transformer.

    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``nn.LayerNorm``.
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        pretrain_img_size: int = 224,
        in_channels: int = 3,
        embed_dims: int = 96,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: int = 4,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        strides: tuple[int, ...] = (4, 2, 2, 2),
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        activation: Callable[..., nn.Module] = nn.GELU,
        normalization: Callable[..., nn.Module] = nn.LayerNorm,
        with_cp: bool = False,
        pretrained: str | None = None,
        convert_weights: bool = False,
        frozen_stages: int = -1,
        init_cfg: dict | None = None,
    ):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])

            if len(pretrain_img_size) != 2:
                msg = f"The size of image should have length 1 or 2, but got {len(pretrain_img_size)}"
                raise ValueError(msg)

        if init_cfg and pretrained:
            msg = "init_cfg and pretrained cannot be set simultaneously"
            raise ValueError(msg)

        init_cfg = {} if init_cfg is None else init_cfg
        if isinstance(pretrained, str):
            warnings.warn("DeprecationWarning: pretrained is deprecated, please use init_cfg instead", stacklevel=2)
            self.init_cfg = {"type": "Pretrained", "checkpoint": pretrained}
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            msg = "pretrained must be a str or None"
            raise TypeError(msg)

        super().__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices

        if strides[0] != patch_size:
            msg = "Use non-overlapping patch embed."
            raise ValueError(msg)

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=strides[0],
            normalization=normalization if patch_norm else None,
            init_cfg=None,
        )

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    normalization=normalization if patch_norm else None,
                    init_cfg=None,
                )
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=downsample,
                activation=activation,
                normalization=normalization,
                with_cp=with_cp,
                init_cfg=None,
            )
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(normalization, self.num_features[i])[1]
            layer_name = f"norm{i}"
            self.add_module(layer_name, layer)

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze stages when training."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f"norm{i-1}")
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self) -> None:
        """Initialize the weights."""
        if not self.init_cfg:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            if "checkpoint" not in self.init_cfg:
                msg = "The checkpoint is not in the init_cfg."
                raise ValueError(msg)
            if not isinstance(self.init_cfg, dict):
                msg = "init_cfg must be a dict"
                raise TypeError(msg)

            ckpt_path = self.init_cfg["checkpoint"]
            if ckpt_path.startswith("http"):
                ckpt = load_from_http(ckpt_path, map_location="cpu")
            elif Path(ckpt_path).exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
            else:
                raise FileNotFoundError(ckpt_path)

            if "state_dict" in ckpt:
                _state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                _state_dict = ckpt["model"]
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = {}
            for k, v in _state_dict.items():
                if k.startswith("backbone."):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if next(iter(state_dict.keys())).startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get("absolute_pos_embed") is not None:
                absolute_pos_embed = state_dict["absolute_pos_embed"]
                n1, length, c1 = absolute_pos_embed.size()
                n2, c2, h, w = self.absolute_pos_embed.size()
                if n1 != n2 or c1 != c2 or h * w != length:
                    warnings.warn("Error in loading absolute_pos_embed, pass", stacklevel=2)
                else:
                    state_dict["absolute_pos_embed"] = (
                        absolute_pos_embed.view(n2, h, w, c2).permute(0, 3, 1, 2).contiguous()
                    )

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [k for k in state_dict if "relative_position_bias_table" in k]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                l1, n_h1 = table_pretrained.size()
                l2, n_h2 = table_current.size()
                if n_h1 != n_h2:
                    warnings.warn(f"Error in loading {table_key}, pass", stacklevel=2)
                elif l1 != l2:
                    s1 = int(l1**0.5)
                    s2 = int(l2**0.5)
                    table_pretrained_resized = torch.nn.functional.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, n_h1, s1, s1),
                        size=(s2, s2),
                        mode="bicubic",
                    )
                    state_dict[table_key] = table_pretrained_resized.view(n_h2, l2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward function."""
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs


def swin_converter(ckpt: dict) -> OrderedDict:
    """Convert the key of pre-trained model from original repo."""
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x: Tensor) -> Tensor:
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        return x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)

    def correct_unfold_norm_order(x: Tensor) -> Tensor:
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        return x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)

    for k, v in ckpt.items():
        if k.startswith("head"):
            continue
        if k.startswith("layers"):
            new_v = v
            if "attn." in k:
                new_k = k.replace("attn.", "attn.w_msa.")
            elif "mlp." in k:
                if "mlp.fc1." in k:
                    new_k = k.replace("mlp.fc1.", "ffn.layers.0.0.")
                elif "mlp.fc2." in k:
                    new_k = k.replace("mlp.fc2.", "ffn.layers.1.")
                else:
                    new_k = k.replace("mlp.", "ffn.")
            elif "downsample" in k:
                new_k = k
                if "reduction." in k:
                    new_v = correct_unfold_reduction_order(v)
                elif "norm." in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace("layers", "stages", 1)
        elif k.startswith("patch_embed"):
            new_v = v
            new_k = k.replace("proj", "projection") if "proj" in k else k
        else:
            new_v = v
            new_k = k

        new_ckpt["backbone." + new_k] = new_v

    return new_ckpt
