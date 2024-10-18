# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of common transformer layers."""

from __future__ import annotations

import copy
import math
from typing import Callable

import torch
from otx.algo.common.utils.utils import get_clones
from otx.algo.modules.transformer import deformable_attention_core_func
from torch import Tensor, nn
from torch.nn import init


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
            src = self.self_attn(q, k, value=src, key_padding_mask=src_mask)[0]
        else:
            src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

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


class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module.

    Args:
        embed_dim (int): The number of expected features in the input.
        num_heads (int): The number of heads in the multiheadattention models.
        num_levels (int): The number of levels in MSDeformableAttention.
        num_points (int): The number of points in MSDeformableAttention.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_levels: int = 4, num_points: int = 4) -> None:
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
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values  # noqa: PD011
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
