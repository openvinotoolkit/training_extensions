# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskDINO transformer decoder module."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

from otx.algo.common.utils.utils import inverse_sigmoid
from otx.algo.detection.heads.rtdetr_decoder import MSDeformableAttention as MSDeformAttn
from otx.algo.instance_segmentation.utils.utils import (
    MLP,
    gen_sineembed_for_position,
    get_clones,
)


class TransformerDecoder(nn.Module):
    """Transformer decoder module."""

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module,
        d_model: int = 256,
        query_dim: int = 4,
    ):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.query_dim = query_dim
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.d_model = d_model
        self.bbox_embed = None
        self.class_embed = None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()  # noqa: SLF001

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_key_padding_mask: Tensor,
        refpoints_unsigmoid: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
    ) -> list[list[Tensor]]:
        """Forward pass."""
        output = tgt
        device = tgt.device

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            reference_points_input = (
                reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )  # nq, bs, nlevel, 4
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # nq, bs, 256*2

            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            output = layer(
                tgt=output,
                tgt_query_pos=raw_query_pos,
                tgt_reference_points=reference_points_input,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_spatial_shapes=spatial_shapes,
                self_attn_mask=tgt_mask,
            )

            # iter update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output).to(device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable transformer decoder layer module."""

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor) -> Tensor:
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: Tensor) -> Tensor:
        """Forward pass for feed forward network."""
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    @autocast(enabled=False)
    def forward(
        self,
        # for tgt
        tgt: Tensor,
        tgt_query_pos: Tensor,
        tgt_reference_points: Tensor,
        # for memory
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        memory_spatial_shapes: Tensor,
        self_attn_mask: Tensor,
    ) -> Tensor:
        """Forward pass."""
        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1),
            memory_spatial_shapes,
            memory_key_padding_mask,
        ).transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        return self.forward_ffn(tgt)
