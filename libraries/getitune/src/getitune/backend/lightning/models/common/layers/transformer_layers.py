# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Implementation of common transformer layers."""

from __future__ import annotations

import copy
import math
from functools import partial
from typing import Any, Callable, NoReturn

import torch
import torch.nn.functional as f
import torchvision
from torch import Tensor, nn
from torch.nn import init

from getitune.backend.lightning.models.common.utils.utils import get_clones, inverse_sigmoid
from getitune.backend.lightning.models.modules.norm import RMSNorm
from getitune.backend.lightning.models.modules.transformer import (
    deformable_attention_core_func,
)
from getitune.backend.lightning.models.utils.weight_init import bias_init_with_prob


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.GELU,
        normalize_before: bool = False,
        batch_first: bool = True,
        key_mask: bool = False,
    ) -> None:
        super().__init__()
        self.normalize_before = normalize_before
        self.key_mask = key_mask

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: torch.Tensor | None) -> torch.Tensor:
        """Attach position embeddings to the tensor."""
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward the transformer encoder layer.

        Args:
            src (torch.Tensor): The input tensor.
            src_mask (torch.Tensor | None, optional): The mask tensor. Defaults to None.
            pos_embed (torch.Tensor | None, optional): The position embedding tensor. Defaults to None.
        """
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        if self.key_mask:
            src, _ = self.self_attn(q, k, value=src, key_padding_mask=src_mask, need_weights=False)
        else:
            src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask, need_weights=False)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class ListForwardMixin:
    """Mixin class that provides list-based forward operations for transformers."""

    def forward(self, x: Tensor) -> NoReturn:
        """Forward pass - must be implemented by subclass."""
        raise NotImplementedError

    def forward_list(self, x_list: list[Tensor]) -> list[Tensor]:
        """Process a list of tensors by concatenating, forwarding, and splitting.

        Args:
            x_list: List of input tensors.

        Returns:
            List of processed tensors with original shapes.
        """
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class LayerScale(nn.Module):
    """Learnable per-channel scaling layer for transformer blocks.

    Args:
        dim: Number of channels/features.
        init_values: Initial scale value.
        inplace: If True, apply scaling in-place.
        device: Device for parameters.
    """

    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self) -> None:
        """Reset gamma parameter to initial value."""
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        """Apply learnable scaling to input tensor."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerEncoder(nn.Module):
    """TransformerEncoder."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None) -> None:
        """Initialize the TransformerEncoder.

        Args:
            encoder_layer (nn.Module): The encoder layer module.
            num_layers (int): The number of layers.
            norm (nn.Module | None, optional): The normalization module. Defaults to None.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward the transformer encoder.

        Args:
            src (torch.Tensor): The input tensor.
            src_mask (torch.Tensor | None, optional): The mask tensor. Defaults to None.
            pos_embed (torch.Tensor | None, optional): The position embedding tensor. Defaults to None.
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MLP(nn.Module):
    """A classic Multi Layer Perceptron (MLP).

    Args:
        input_dim (int): The number of expected features in the input.
        hidden_dim (int): The number of features in the hidden layers.
        output_dim (int): The number of features in the output layer.
        num_layers (int): The number of layers in the MLP.
        activation (Callable[..., nn.Module] | None, optional): The activation function. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.act = activation() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of MLP."""
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MLP2L(nn.Module, ListForwardMixin):
    """Multi-Layer Perceptron for Vision Transformer with 2 fixed layers.

    A simple two-layer MLP with configurable hidden dimension and activation.

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features. Defaults to in_features.
        out_features: Number of output features. Defaults to in_features.
        act_layer: Activation layer class.
        drop: Dropout rate.
        bias: Whether to use bias in linear layers.
        device: Device to place tensors on.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module.

    Args:
        embed_dim (int): The number of expected features in the input.
        num_heads (int): The number of heads in the multiheadattention models.
        num_levels (int): The number of levels in MSDeformableAttention.
        num_points (int): The number of points in MSDeformableAttention.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
    ) -> None:
        """Multi-Scale Deformable Attention Module."""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            msg = f"embed_dim must be divisible by num_heads, but got embed_dim={embed_dim} and num_heads={num_heads}"
            raise ValueError(msg)

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward function of MSDeformableAttention.

        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (list[tuple[int, int]]): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ...]
            value_mask (Tensor | None, optional): [bs, value_length], True for non-padding elements,
                False for padding elements. Defaults to None.

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.reshape(bs, len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs,
            len_q,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query).reshape(
            bs,
            len_q,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        attention_weights = nn.functional.softmax(attention_weights, dim=-1).reshape(
            bs,
            len_q,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = (
                value_spatial_shapes
                if isinstance(value_spatial_shapes, torch.Tensor)
                else torch.tensor(value_spatial_shapes)
            ).clone()
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(
                    bs,
                    len_q,
                    1,
                    self.num_levels,
                    1,
                    2,
                )
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        elif reference_points.shape[-1] == 6:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * (reference_points[:, :, None, :, None, 2::2] + reference_points[:, :, None, :, None, 3::2])
                * 0.5
            )
        else:
            msg = f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            raise ValueError(
                msg,
            )

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        return self.output_proj(output)


class MSDeformableAttentionV2(nn.Module):
    """Multi-Scale Deformable Attention Module V2.

    Note:
        This is different from vanilla MSDeformableAttention where it uses
        distinct number of sampling points for features at different scales.
        Refer to RTDETRv2.

    Args:
        embed_dim (int): The number of expected features in the input.
        num_heads (int): The number of heads in the multiheadattention models.
        num_levels (int): The number of levels in MSDeformableAttention.
        num_points_list (list[int]): Number of distinct points for each layer. Defaults to [3, 6, 3].
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points_list: list[int] = [3, 6, 3],  # noqa: B006
        method: str = "default",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points_list = num_points_list
        self.method = method

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale",
            torch.tensor(num_points_scale, dtype=torch.float32),
        )

        self.total_points = num_heads * sum(num_points_list)
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self) -> None:
        """Reset parameters of the model."""
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        value_spatial_shapes: list[list[int]],
    ) -> Tensor:
        """Forward function of MSDeformableAttention.

        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        _, n_head, c, _ = value[0].shape
        num_points_list = self.num_points_list

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs,
            len_q,
            self.num_heads,
            sum(self.num_points_list),
            2,
        )

        attention_weights = self.attention_weights(query).reshape(
            bs,
            len_q,
            self.num_heads,
            sum(self.num_points_list),
        )
        attention_weights = f.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(
                    bs,
                    len_q,
                    1,
                    self.num_levels,
                    1,
                    2,
                )
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(query).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * 0.5
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            msg = (f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.",)
            raise ValueError(msg)

        # sampling_offsets [8, 480, 8, 12, 2]
        sampling_grids = 2 * sampling_locations - 1

        sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

        sampling_value_list = []
        for level, (h, w) in enumerate(value_spatial_shapes):
            value_l = value[level].reshape(bs * n_head, c, h, w)
            sampling_grid_l = sampling_locations_list[level]
            sampling_value_l = f.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            sampling_value_list.append(sampling_value_l)

        attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, len_q, sum(num_points_list))
        weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
        output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, len_q)

        return output.permute(0, 2, 1)


class VisualEncoderLayer(nn.Module):
    """VisualEncoderLayer module consisting of MSDeformableAttention and feed-forward network.

    Args:
        d_model (int): The input and output dimension of the layer. Defaults to 256.
        d_ffn (int): The hidden dimension of the feed-forward network. Defaults to 1024.
        dropout (float): The dropout rate. Defaults to 0.1.
        activation (Callable[..., nn.Module]): The activation function. Defaults to nn.ReLU.
        n_levels (int): The number of feature levels. Defaults to 4.
        n_heads (int): The number of attention heads. Defaults to 8.
        n_points (int): The number of sampling points for the MSDeformableAttention. Defaults to 4.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ) -> None:
        super().__init__()

        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        """Add position embedding to the input tensor.

        Args:
            tensor (Tensor): The input tensor.
            pos (Tensor | None): The position embedding tensor. Defaults to None.

        Returns:
            Tensor: The tensor with position embedding added.
        """
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: Tensor) -> Tensor:
        """Forward pass of the ffn.

        Args:
            src (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        reference_points: Tensor,
        spatial_shapes: list[tuple[int, int]],
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the VisualEncoderLayer.

        Args:
            src (Tensor): The input tensor.
            pos (Tensor): The position embedding tensor.
            reference_points (Tensor): The reference points tensor.
            spatial_shapes (List[Tuple[int, int]]): The list of spatial shapes.
            padding_mask (Optional[Tensor]): The padding mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        return self.forward_ffn(src)


class VisualEncoder(nn.Module):
    """VisualEncoder module consisting of multiple VisualEncoderLayer modules.

    Args:
        encoder_layer (VisualEncoderLayer): The Visual encoder layer module.
        num_layers (int): The number of layers.
    """

    def __init__(self, encoder_layer: VisualEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(
        spatial_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Generate reference points for each spatial level.

        Args:
            spatial_shapes (List[Tuple[int, int]]): The list of spatial shapes.
            valid_ratios (Tensor): The tensor of valid ratios.
            device (torch.device): The device to use.

        Returns:
            Tensor: The tensor of reference points.
        """
        reference_points_list = []
        for lvl, (h_, w_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h_ - 0.5, h_, device=device),
                torch.linspace(0.5, w_ - 0.5, w_, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        spatial_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        pos: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the VisualEncoder module.

        Args:
            src (Tensor): The input tensor.
            spatial_shapes (List[Tuple[int, int]]): The list of spatial shapes.
            level_start_index (Tensor): The level start index tensor.
            valid_ratios (Tensor): The tensor of valid ratios.
            pos (Tensor | None): The position embedding tensor. Defaults to None.
            padding_mask (Tensor | None): The padding mask tensor. Defaults to None.
            ref_token_index (int | None): The reference token index. Defaults to None.
            ref_token_coord (Tensor | None): The reference token coordinates. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, padding_mask)

        return output


def cat_keep_shapes(x_list: list[Tensor]) -> tuple[Tensor, list[tuple[int, ...]], list[int]]:
    """Concatenate tensors while preserving their original shapes.

    Args:
        x_list: List of tensors to concatenate.

    Returns:
        Tuple of (flattened tensor, original shapes, token counts).
    """
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: list[tuple[int, ...]], num_tokens: list[int]) -> list[Tensor]:
    """Split a flattened tensor back to original shapes.

    Args:
        flattened: Concatenated tensor.
        shapes: Original tensor shapes.
        num_tokens: Token counts for splitting.

    Returns:
        List of tensors with original shapes.
    """
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    return [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the tensor elements for RoPE.

    Args:
        x: Input tensor of shape [..., D].

    Returns:
        Rotated tensor where x[..., :D/2] and x[..., D/2:] are swapped and negated.
    """
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Apply rotary position embedding to tensor.

    Args:
        x: Input tensor of shape [..., D].
        sin: Sine embeddings of shape [..., D].
        cos: Cosine embeddings of shape [..., D].

    Returns:
        Tensor with rotary position embedding applied.
    """
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    """Linear layer with masked bias for Q, K, V projection.

    Masks the K bias portion with NaN values for specific attention patterns.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        o = self.out_features
        if o % 3 != 0:
            msg = f"out_features ({o}) must be divisible by 3"
            raise ValueError(msg)
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        """Apply linear transformation with masked bias.

        Args:
            input: Input tensor.

        Returns:
            Transformed tensor.
        """
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return f.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    """Multi-head self-attention module.

    Args:
        dim: Input/output feature dimension.
        num_heads: Number of attention heads.
        qkv_bias: If True, add bias to QKV projection.
        proj_bias: If True, add bias to output projection.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
        mask_k_bias: If True, mask the K bias.
        device: Device for parameters.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape [B, heads, N, D//heads].
            k: Key tensor of shape [B, heads, N, D//heads].
            rope: Tuple of (sin, cos) tensors for position embedding.

        Returns:
            Tuple of (q, k) with rotary embeddings applied.
        """
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        n = q.shape[-2]
        prefix = n - sin.shape[-2]
        if prefix < 0:
            msg = f"prefix ({prefix}) must be >= 0"
            raise ValueError(msg)
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor | None = None,
        rope: Tensor | tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Forward pass for self-attention.

        Args:
            x: Input tensor of shape [B, N, D].
            attn_bias: Optional attention bias.
            rope: Optional rotary position embedding.

        Returns:
            Output tensor of shape [B, N, D].
        """
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        return self.proj_drop(x)

    def forward_list(
        self,
        x_list: list[Tensor],
        attn_bias: Tensor | None = None,
        rope_list: list[tuple[Tensor, Tensor]] | None = None,
    ) -> list[Tensor]:
        """Forward pass for list of tensors.

        Args:
            x_list: List of input tensors.
            attn_bias: Optional attention bias.
            rope_list: List of rotary position embeddings.

        Returns:
            List of output tensors.
        """
        if rope_list is None or len(x_list) != len(rope_list):
            msg = "x_list and rope_list must have same length"
            raise ValueError(msg)
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(
        self,
        qkv: Tensor,
        attn_bias: Tensor | None = None,
        rope: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Compute attention from QKV tensor.

        Args:
            qkv: Combined query-key-value tensor.
            attn_bias: Optional attention bias (must be None).
            rope: Optional rotary position embedding.

        Returns:
            Attention output tensor.
        """
        if attn_bias is not None:
            msg = "attn_bias must be None"
            raise ValueError(msg)
        B, N, _ = qkv.shape  # noqa: N806
        C = self.qkv.in_features  # noqa: N806

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class SelfAttentionBlock(nn.Module):
    """Transformer block with self-attention and FFN.

    Args:
        dim: Input/output feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: If True, add bias to QKV projection.
        proj_bias: If True, add bias to output projection.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        init_values: Initial values for LayerScale.
        drop_path: Drop path rate.
        act_layer: Activation layer class.
        norm_layer: Normalization layer class.
        rope_subset_list: List of RoPE subsets.
        ffn_layer: FFN layer class.
        mask_k_bias: If True, mask the K bias.
        device: Device for parameters.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = MLP2L,
        mask_k_bias: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        if sin.ndim != cos.ndim:
            msg = "sin and cos must have same ndim"
            raise ValueError(msg)
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        # No batch dimension, do not index
        return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """Forward pass for a single tensor.

        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: list[Tensor], rope_list: list[tuple[Tensor, Tensor]] | None = None) -> list[Tensor]:
        """Forward pass for list of tensors.

        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list: list[tuple[Tensor, Tensor] | None] | None = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = None

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for i, x in enumerate(x_list):
                rope = rope_list[i] if rope_list is not None else None
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(
        self,
        x_or_x_list: Tensor | list[Tensor],
        rope_or_rope_list: tuple[Tensor, Tensor] | list[tuple[Tensor, Tensor] | None] | None = None,
    ) -> Tensor | list[Tensor]:
        """Forward pass supporting both single tensor and list of tensors.

        Args:
            x_or_x_list: Input tensor or list of tensors.
            rope_or_rope_list: Rotary position embedding or list of embeddings.

        Returns:
            Output tensor or list of tensors.
        """
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            rope_as_list = [rope_or_rope_list] if not isinstance(rope_or_rope_list, list) else rope_or_rope_list
            return self._forward_list([x_or_x_list], rope_list=rope_as_list)[0]  # type: ignore[arg-type]
        if isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _ in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)  # type: ignore[arg-type]
        msg = f"x_or_x_list must be Tensor or list, got {type(x_or_x_list)}"
        raise TypeError(msg)


class Gate(nn.Module):
    """Gated fusion module for combining two feature streams.

    Uses learnable gates to adaptively blend two input tensors.

    Args:
        d_model: Feature dimension.
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm.
    """

    def __init__(self, d_model: int, use_rmsnorm: bool = False) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Gated fusion of two tensors.

        Args:
            x1: First input tensor of shape (B, N, C).
            x2: Second input tensor of shape (B, N, C).

        Returns:
            Fused tensor of shape (B, N, C).
        """
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """Integral layer for distribution-based bounding box regression.

    Computes target location using: `sum{Pr(n) * W(n)}`, where Pr(n) is the
    softmax probability vector and W(n) is the non-uniform weighting function.

    Args:
        reg_max: Maximum number of discrete bins for regression.
    """

    def __init__(self, reg_max: int = 32) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x: Tensor, project: Tensor) -> Tensor:
        """Compute integral over distribution.

        Args:
            x: Distribution tensor of shape (B, N, 4*(reg_max+1)).
            project: Projection weights for weighted sum.

        Returns:
            Bounding box offsets of shape (B, N, 4).
        """
        shape = x.shape
        x = f.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = f.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape([*list(shape[:-1]), -1])


class LQE(nn.Module):
    """Location Quality Estimator.

    Estimates localization quality from corner distribution statistics
    to refine classification scores.

    Args:
        k: Number of top probabilities to use for statistics.
        hidden_dim: Hidden dimension for MLP.
        num_layers: Number of MLP layers.
        reg_max: Maximum regression bins.
        activation: Activation function class.
    """

    def __init__(
        self,
        k: int,
        hidden_dim: int,
        num_layers: int,
        reg_max: int,
        activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
    ) -> None:
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(
            input_dim=4 * (k + 1),
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation=activation,
        )
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores: Tensor, pred_corners: Tensor) -> Tensor:
        """Refine scores based on corner distribution quality.

        Args:
            scores: Classification scores of shape (B, N, num_classes).
            pred_corners: Corner predictions of shape (B, N, 4*(reg_max+1)).

        Returns:
            Refined scores of shape (B, N, num_classes).
        """
        b, num_pred, _ = pred_corners.size()
        prob = f.softmax(pred_corners.reshape(b, num_pred, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(b, num_pred, -1))
        return scores + quality_score


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Implements the SwiGLU activation function as described in GLU Variants paper.
    Uses gated linear units with SiLU activation for improved performance.

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features.
        out_features: Number of output features.
        bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weights with Xavier uniform and zero bias."""
        init.xavier_uniform_(self.w12.weight)
        init.constant_(self.w12.bias, 0)
        init.xavier_uniform_(self.w3.weight)
        init.constant_(self.w3.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, out_features).
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = f.silu(x1) * x2
        return self.w3(hidden)


def get_contrastive_denoising_training_group(
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    num_queries: int,
    class_embed: torch.nn.Module,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    max_denoising_queries: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | tuple[None, None, None, None]:
    """Generate contrastive denoising training group with memory-efficient capping.

    This function creates noisy versions of ground truth boxes for denoising training.
    When there are too many objects per image, it subsamples ground truth to prevent OOM.

    Memory usage scales with (num_denoising_queries + num_queries)² for attention mask.

    Args:
        targets (List[Dict[str, torch.Tensor]]): List of target dictionaries.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.nn.Module): Class embedding module.
        num_denoising (int, optional): Target number of denoising queries (soft hint). Defaults to 100.
        label_noise_ratio (float, optional): Ratio of label noise. Defaults to 0.5.
        box_noise_scale (float, optional): Scale of box noise. Defaults to 1.0.
        max_denoising_queries (int, optional): Hard limit on denoising queries to prevent OOM. Defaults to 1000.

    Returns:
        Tuple[Tensor,Tensor,Tensor, dict[str, Tensor]] | tuple[None,None,None,None]:
        Tuple containing input query class, input query bbox, attention mask, and denoising metadata.
    """
    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group

    # Cap the number of denoising queries to prevent OOM with many ground truth objects
    # Each GT produces 2 queries (positive + negative) per group
    total_dn_queries = max_gt_num * 2 * num_group
    if total_dn_queries > max_denoising_queries:
        # First, try reducing the number of groups
        num_group = max_denoising_queries // (max_gt_num * 2)

        if num_group < 1:
            # Even with 1 group, max_gt_num * 2 exceeds limit
            # Must subsample ground truth boxes
            num_group = 1
            max_gt_num_capped = max_denoising_queries // 2

            # Vectorized subsampling with consistent permutation for labels and boxes
            def subsample_target(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                n = len(t["labels"])
                if n > max_gt_num_capped:
                    return {"labels": t["labels"][:max_gt_num_capped], "boxes": t["boxes"][:max_gt_num_capped]}
                return t

            targets = [subsample_target(t) for t in targets]
            num_gts = [min(len(t["labels"]), max_gt_num_capped) for t in targets]
            max_gt_num = max_gt_num_capped

    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = torchvision.ops.box_convert(input_query_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = torchvision.ops.box_convert(known_bbox, in_fmt="xyxy", out_fmt="cxcywh")
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    input_query_class = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1) : num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1) : num_denoising] = True
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
        "dn_num_gts": num_gts,
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta
