# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskDINO transformer decoder module."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

from otx.algo.detection.heads.rtdetr_decoder import MSDeformableAttention as MSDeformAttn
from otx.algo.instance_segmentation.mask_dino.utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_sineembed_for_position,
)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module,
        d_model: int = 256,
        query_dim: int = 4,
        dec_layer_share: bool = False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        self.norm = norm
        self.query_dim = query_dim
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        refpoints_unsigmoid: Tensor | None = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Tensor | None = None,  # num_levels
        spatial_shapes: Tensor | None = None,  # bs, num_levels, 2
        valid_ratios: Tensor | None = None,
    ):
        """Input:
        - tgt: nq, bs, d_model
        - memory: hw, bs, d_model
        - pos: hw, bs, d_model
        - refpoints_unsigmoid: nq, bs, 2/4
        - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt
        device = tgt.device

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]

        for layer in self.layers:
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

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_deformable_box_attn=False,
        key_aware_type=None,
    ):
        super().__init__()

        # cross attention
        if use_deformable_box_attn:
            raise NotImplementedError
        else:
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
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @autocast(enabled=False)
    def forward(
        self,
        # for tgt
        tgt: Tensor | None,  # nq, bs, d_model
        tgt_query_pos: Tensor | None = None,  # pos for query. MLP(Sine(pos))
        tgt_reference_points: Tensor | None = None,  # nq, bs, 4
        # for memory
        memory: Tensor | None = None,  # hw, bs, d_model
        memory_key_padding_mask: Tensor | None = None,
        memory_spatial_shapes: Tensor | None = None,  # bs, num_levels, 2
        self_attn_mask: Tensor | None = None,  # mask used for self-attention
    ):
        """Input:
        - tgt/tgt_query_pos: nq, bs, d_model
        -
        """
        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # cross attention
        if self.key_aware_type is not None:
            if self.key_aware_type == "mean":
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == "proj_mean":
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError(f"Unknown key_aware_type: {self.key_aware_type}")
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
        tgt = self.forward_ffn(tgt)

        return tgt
