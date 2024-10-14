# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""depth aware transformer head for 3d object detection."""
from __future__ import annotations

import math
from typing import Any, Callable, ClassVar

import torch
from torch import Tensor, nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from otx.algo.detection.heads.rtdetr_decoder import MLP, MSDeformableAttention
from otx.algo.detection.utils.utils import inverse_sigmoid
from otx.algo.object_detection_3d.utils.utils import get_clones


def gen_sineembed_for_position(pos_tensor: Tensor) -> Tensor:
    """Generate sine embeddings for position tensor.

    Args:
        pos_tensor (Tensor): Position tensor of shape (n_query, bs, num_dims).

    Returns:
        Tensor: Sine embeddings for position tensor of shape (n_query, bs, embedding_dim).
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 6:
        for i in range(2, 6):  # Compute sine embeds for l, r, t, b
            embed = pos_tensor[:, :, i] * scale
            pos_embed = embed[:, :, None] / dim_t
            pos_embed = torch.stack((pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim=3).flatten(2)
            pos = pos_embed if i == 2 else torch.cat((pos, pos_embed), dim=2)
        pos = torch.cat((pos_y, pos_x, pos), dim=2)
    else:
        msg = f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}"
        raise ValueError(msg)
    return pos


class DepthAwareTransformer(nn.Module):
    """DepthAwareTransformer module."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        return_intermediate_dec: bool = False,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        enc_n_points: int = 4,
        group_num: int = 11,
    ) -> None:
        """Initialize the DepthAwareTransformer module.

        Args:
            d_model (int): The dimension of the input and output feature vectors.
            nhead (int): The number of attention heads.
            num_encoder_layers (int): The number of encoder layers.
            num_decoder_layers (int): The number of decoder layers.
            dim_feedforward (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
            activation (Callable[..., nn.Module]): The activation function.
            return_intermediate_dec (bool): Whether to return intermediate decoder outputs.
            num_feature_levels (int): The number of feature levels.
            dec_n_points (int): The number of points for the decoder attention.
            enc_n_points (int): The number of points for the encoder attention.
            group_num (int): The number of groups for the two-stage training.
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.group_num = group_num

        encoder_layer = VisualEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = VisualEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DepthAwareDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            group_num=group_num,
        )
        self.decoder = DepthAwareDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate_dec,
            d_model,
            activation,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters of the model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention):
                m._reset_parameters()  # noqa: SLF001
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals: Tensor) -> Tensor:
        """Generate position embeddings for proposal tensor.

        Args:
            proposals (Tensor): Proposal tensor of shape (N, L, 6).

        Returns:
            Tensor: Position embeddings for proposal tensor of shape (N, L, embedding_dim).
        """
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 6
        proposals = proposals.sigmoid() * scale
        # N, L, 6, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 6, 64, 2
        return torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)

    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        """Calculate the valid ratio of the mask.

        Args:
            mask (Tensor): The mask tensor.

        Returns:
            Tensor: The valid ratio tensor.
        """
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(
        self,
        srcs: list[Tensor],
        masks: list[Tensor],
        pos_embeds: list[Tensor],
        query_embed: Tensor,
        depth_pos_embed: Tensor,
        depth_pos_embed_ip: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Forward pass of the DepthAwareTransformer module.

        Args:
            srcs (List[Tensor]): List of source tensors.
            masks (List[Tensor]): List of mask tensors.
            pos_embeds (List[Tensor]): List of position embedding tensors.
            query_embed (Tensor | None): Query embedding tensor. Defaults to None.
            depth_pos_embed (Tensor | None): Depth position embedding tensor. Defaults to None.
            depth_pos_embed_ip (Tensor | None): Depth position embedding IP tensor. Defaults to None.
            attn_mask (Tensor | None): Attention mask tensor. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]: Tuple containing the output tensors.
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes_list = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_list.append(spatial_shape)
            src_ = src.flatten(2).transpose(1, 2)
            pos_embed_ = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed_ + self.level_embed[lvl].view(1, 1, -1)

            mask_ = mask.flatten(1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src_)
            mask_flatten.append(mask_)

        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=srcs[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )
        # enc_intermediate_output, enc_intermediate_refpoints = None
        # prepare input for decoder
        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        depth_pos_embed = depth_pos_embed.flatten(2).permute(2, 0, 1)
        depth_pos_embed_ip = depth_pos_embed_ip.flatten(2).permute(2, 0, 1)
        mask_depth = masks[1].flatten(1)

        # decoder
        # ipdb.set_trace()
        hs, inter_references, inter_references_dim = self.decoder(
            tgt,  # .transpose(1,0), for DINO
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,  # ,INFo
            mask_flatten,
            depth_pos_embed,
            mask_depth,
            bs=bs,
            depth_pos_embed_ip=depth_pos_embed_ip,
            pos_embeds=pos_embeds,
            attn_mask=attn_mask,
        )

        inter_references_out = inter_references
        inter_references_out_dim = inter_references_dim
        return hs, init_reference_out, inter_references_out, inter_references_out_dim, None, None


class VisualEncoderLayer(nn.Module):
    """VisualEncoderLayer module."""

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
        """Initialize the DepthAwareDecoderLayer.

        Args:
            d_model (int): The input and output dimension of the layer. Defaults to 256.
            d_ffn (int): The hidden dimension of the feed-forward network. Defaults to 1024.
            dropout (float): The dropout rate. Defaults to 0.1.
            activation (Callable[..., nn.Module]): The activation function. Defaults to nn.ReLU.
            n_levels (int): The number of feature levels. Defaults to 4.
            n_heads (int): The number of attention heads. Defaults to 8.
            n_points (int): The number of sampling points for the MSDeformableAttention. Defaults to 4.
        """
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
        level_start_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the VisualEncoderLayer.

        Args:
            src (Tensor): The input tensor.
            pos (Tensor): The position embedding tensor.
            reference_points (Tensor): The reference points tensor.
            spatial_shapes (List[Tuple[int, int]]): The list of spatial shapes.
            level_start_index (Tensor): The level start index tensor.
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
    """VisualEncoder module."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        """Initialize the DepthAwareDecoder.

        Args:
            encoder_layer (nn.Module): The encoder layer module.
            num_layers (int): The number of layers.
        """
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
                torch.linspace(0.5, h_ - 0.5, h_, dtype=torch.float32, device=device),
                torch.linspace(0.5, w_ - 0.5, w_, dtype=torch.float32, device=device),
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
        level_start_index: Tensor,
        valid_ratios: Tensor,
        pos: Tensor | None = None,
        padding_mask: Tensor | None = None,
        ref_token_index: int | None = None,
        ref_token_coord: Tensor | None = None,
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
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DepthAwareDecoderLayer(nn.Module):
    """DepthAwareDecoderLayer module."""

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        group_num: int = 1,
    ) -> None:
        """Initialize the DepthAwareDecoderLayer.

        Args:
            d_model (int): The input and output dimension of the layer. Defaults to 256.
            d_ffn (int): The hidden dimension of the feed-forward network. Defaults to 1024.
            dropout (float): The dropout rate. Defaults to 0.1.
            activation (Callable[..., nn.Module]): The activation function. Defaults to nn.ReLU.
            n_levels (int): The number of feature levels. Defaults to 4.
            n_heads (int): The number of attention heads. Defaults to 8.
            n_points (int): The number of sampling points for the MSDeformableAttention. Defaults to 4.
            group_num (int): The number of groups for training. Defaults to 1.
        """
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # depth cross attention
        self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.group_num = group_num

        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.nhead = n_heads

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

    def forward_ffn(self, tgt: Tensor) -> Tensor:
        """Forward pass of the ffn.

        Args:
            tgt (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        tgt: Tensor,
        query_pos: Tensor,
        reference_points: Tensor,
        src: Tensor,
        src_spatial_shapes: list[tuple[int, int]],
        level_start_index: Tensor,
        src_padding_mask: Tensor,
        depth_pos_embed: Tensor,
        mask_depth: Tensor,
        bs: int,
        query_sine_embed: Tensor | None = None,
        is_first: bool | None = None,
        depth_pos_embed_ip: Tensor | None = None,
        pos_embeds: list[Tensor] | None = None,
        self_attn_mask: Tensor | None = None,
        query_pos_un: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the DepthAwareDecoder module.

        Args:
            tgt (Tensor): The input tensor.
            query_pos (Tensor): The query position tensor.
            reference_points (Tensor): The reference points tensor.
            src (Tensor): The source tensor.
            src_spatial_shapes (List[Tuple[int, int]]): The list of spatial shapes.
            level_start_index (Tensor): The level start index tensor.
            src_padding_mask (Tensor): The source padding mask tensor.
            depth_pos_embed (Tensor): The depth position embedding tensor.
            mask_depth (Tensor): The depth mask tensor.
            bs (int): The batch size.
            query_sine_embed (Tensor | None): The query sine embedding tensor. Defaults to None.
            is_first (bool | None): Whether it is the first iteration. Defaults to None.
            depth_pos_embed_ip (Tensor | None): The depth position embedding tensor for the iterative process.
                Defaults to None.
            pos_embeds (List[Tensor] | None): The list of position embedding tensors. Defaults to None.
            self_attn_mask (Tensor | None): The self-attention mask tensor. Defaults to None.
            query_pos_un (Tensor | None): The unnormalized query position tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        # depth cross attention
        tgt2 = self.cross_attn_depth(
            tgt.transpose(0, 1),
            depth_pos_embed,
            depth_pos_embed,
            key_padding_mask=mask_depth,
        )[0].transpose(0, 1)

        tgt = tgt + self.dropout_depth(tgt2)
        tgt = self.norm_depth(tgt)

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)

        q_content = self.sa_qcontent_proj(q)
        q_pos = self.sa_qpos_proj(q)
        k_content = self.sa_kcontent_proj(k)
        k_pos = self.sa_kpos_proj(k)
        v = self.sa_v_proj(tgt)
        q = q_content + q_pos
        k = k_content + k_pos

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)
        num_queries = q.shape[0]

        if self.training:
            num_noise = num_queries - self.group_num * 50
            num_queries = self.group_num * 50
            q_noise = q[:num_noise].repeat(1, self.group_num, 1)
            k_noise = k[:num_noise].repeat(1, self.group_num, 1)
            v_noise = v[:num_noise].repeat(1, self.group_num, 1)
            q = q[num_noise:]
            k = k[num_noise:]
            v = v[num_noise:]
            q = torch.cat(q.split(num_queries // self.group_num, dim=0), dim=1)
            k = torch.cat(k.split(num_queries // self.group_num, dim=0), dim=1)
            v = torch.cat(v.split(num_queries // self.group_num, dim=0), dim=1)
            q = torch.cat([q_noise, q], dim=0)
            k = torch.cat([k_noise, k], dim=0)
            v = torch.cat([v_noise, v], dim=0)

        tgt2 = self.self_attn(q, k, v)[0]
        tgt2 = torch.cat(tgt2.split(bs, dim=1), dim=0).transpose(0, 1) if self.training else tgt2.transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return self.forward_ffn(tgt)


class DepthAwareDecoder(nn.Module):
    """DepthAwareDecoder module."""

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        return_intermediate: bool,
        d_model: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        """Initialize the DepthAwareDecoder.

        Args:
            decoder_layer (nn.Module): The decoder layer module.
            num_layers (int): The number of layers.
            return_intermediate (bool, optional): Whether to return intermediate outputs. Defaults to False.
            d_model (int | None, optional): The input and output dimension of the layer. Defaults to None.
        """
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None

        self.query_scale = MLP(d_model, d_model, d_model, 2, activation=activation)
        self.ref_point_head = MLP(d_model, d_model, 2, 2, activation=activation)

    def forward(
        self,
        tgt: Tensor,
        reference_points: Tensor,
        src: Tensor,
        src_spatial_shapes: list[tuple[int, int]],
        src_level_start_index: Tensor,
        src_valid_ratios: Tensor,
        query_pos: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
        depth_pos_embed: Tensor | None = None,
        mask_depth: Tensor | None = None,
        bs: int | None = None,
        depth_pos_embed_ip: Tensor | None = None,
        pos_embeds: list[Tensor] | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the DepthAwareDecoder module.

        Args:
            tgt (Tensor): The input tensor.
            reference_points (Tensor): The reference points tensor.
            src (Tensor): The source tensor.
            src_spatial_shapes (List[Tuple[int, int]]): The list of spatial shapes.
            src_level_start_index (Tensor): The level start index tensor.
            src_valid_ratios (Tensor): The tensor of valid ratios.
            query_pos (Tensor | None): The query position tensor. Defaults to None.
            src_padding_mask (Tensor | None): The source padding mask tensor. Defaults to None.
            depth_pos_embed (Tensor | None): The depth position embedding tensor. Defaults to None.
            mask_depth (Tensor | None): The depth mask tensor. Defaults to None.
            bs (int | None): The batch size. Defaults to None.
            depth_pos_embed_ip (Tensor | None): The depth position embedding tensor for the iterative process.
                Defaults to None.
            pos_embeds (List[Tensor] | None): The list of position embedding tensors. Defaults to None.
            attn_mask (Tensor | None): The self-attention mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_dims = []
        bs = src.shape[0]

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    msg = f"Wrong reference_points shape[-1]:{reference_points.shape[-1]}"
                    raise ValueError(msg)

                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            ###conditional
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
                depth_pos_embed,
                mask_depth,
                bs,
                query_sine_embed=None,
                is_first=(lid == 0),
                depth_pos_embed_ip=depth_pos_embed_ip,
                pos_embeds=pos_embeds,
                self_attn_mask=attn_mask,
                query_pos_un=None,
            )

            # implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            reference_dims: Tensor
            if self.dim_embed is not None:
                reference_dims = self.dim_embed[lid](output)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_reference_dims.append(reference_dims)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(
                intermediate_reference_dims,
            )

        return output, reference_points, None


class DepthAwareTransformerBuilder:
    """DepthAwareTransformerBuilder."""

    CFG: ClassVar[dict[str, Any]] = {
        "monodetr_50": {
            "d_model": 256,
            "dropout": 0.1,
            "nhead": 8,
            "dim_feedforward": 256,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "return_intermediate_dec": True,
            "num_feature_levels": 4,
            "dec_n_points": 4,
            "enc_n_points": 4,
        },
    }

    def __new__(cls, model_name: str) -> DepthAwareTransformer:
        """Create the DepthAwareTransformer."""
        return DepthAwareTransformer(**cls.CFG[model_name])
