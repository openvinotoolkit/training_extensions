# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""depth aware transformer head for 3d object detection."""

import copy
import math
from typing import Any, ClassVar, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from otx.algo.detection.heads.rtdetr_decoder import MSDeformableAttention, MLP
from otx.algo.detection.utils.utils import inverse_sigmoid


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
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
            if i == 2:  # Initialize pos for the case of size(-1)=6
                pos = pos_embed
            else:  # Concatenate embeds for l, r, t, b
                pos = torch.cat((pos, pos_embed), dim=2)
        pos = torch.cat((pos_y, pos_x, pos), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos


class DepthAwareTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=50,
        group_num=11,
        use_dab=False,
        two_stage_dino=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab
        self.two_stage_dino = two_stage_dino
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
            use_dab=use_dab,
            two_stage_dino=two_stage_dino,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        elif two_stage_dino:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals * group_num, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
            self.two_stage_wh_embedding = None
            self.enc_out_class_embed = None
            self.enc_out_bbox_embed = None
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab and not self.two_stage_dino:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

            lr = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            tb = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            wh = torch.cat((lr, tb), -1)

            proposal = torch.cat((grid, wh), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        srcs,
        masks,
        pos_embeds,
        query_embed=None,
        depth_pos_embed=None,
        depth_pos_embed_ip=None,
        attn_mask=None,
    ):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            mask = mask.flatten(1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)
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
        ####DINO_pos
        if self.two_stage:
            ###share
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            # enc_outputs_class = self.decoder.class_embed(output_memory)
            # enc_outputs_coord_unact = self.decoder.bbox_embed(output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            ####share_end
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points

            topk_coords_unact_input = torch.cat(
                (topk_coords_unact[..., 0:2], topk_coords_unact[..., 2::2] + topk_coords_unact[..., 3::2]),
                dim=-1,
            )

            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact_input)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        elif self.use_dab:
            reference_points = query_embed[..., self.d_model :].sigmoid()
            tgt = query_embed[..., : self.d_model]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            ##for dn
            init_reference_out = reference_points
        elif self.two_stage_dino:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid
            if self.training:
                topk = self.two_stage_num_proposals * self.group_num
            else:
                topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 6),
            )  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 6),
            ).sigmoid()  # sigmoid
            if self.training:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1)  # nq, bs, d_model
            else:
                tgt_ = self.tgt_embed.weight[: self.two_stage_num_proposals, None, :].repeat(
                    1,
                    bs,
                    1,
                )  # nq, bs, d_model
            reference_points, tgt = refpoint_embed_, tgt_
            init_reference_out = reference_points
        else:
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
            query_embed if (not self.use_dab and not self.two_stage_dino) else None,  # ,INFo
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

        if self.two_stage:
            return (
                hs,
                init_reference_out,
                inter_references_out,
                inter_references_out_dim,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        return hs, init_reference_out, inter_references_out, inter_references_out_dim, None, None


class VisualEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation: Callable[..., nn.Module] = nn.ReLU, n_levels=4, n_heads=8, n_points=4):
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
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        # src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class VisualEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
        ref_token_index=None,
        ref_token_coord=None,
    ):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DepthAwareDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        n_levels=4,
        n_heads=8,
        n_points=4,
        group_num=1,
    ):
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
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask,
        depth_pos_embed,
        mask_depth,
        bs,
        query_sine_embed=None,
        is_first=None,
        depth_pos_embed_ip=None,
        pos_embeds=None,
        self_attn_mask=None,
        query_pos_un=None,
    ):
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
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=1), dim=0).transpose(0, 1)

        else:
            tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
        #                        reference_points,
        #                        src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DepthAwareDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        return_intermediate=False,
        d_model=None,
        use_dab=False,
        two_stage_dino=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.two_stgae_dino = two_stage_dino
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2, activation=nn.ReLU)
            self.query_scale_bbox = MLP(d_model, 2, 2, 2)
            self.ref_point_head = MLP(3 * d_model, d_model, d_model, 2, activation=nn.ReLU)
        elif two_stage_dino:
            self.ref_point_head = MLP(3 * d_model, d_model, d_model, 2, activation=nn.ReLU)
            # self.query_scale = None
            self.query_scale = MLP(d_model, d_model, d_model, 2, activation=nn.ReLU)
            self.query_pos_sine_scale = None
            self.ref_anchor_head = None
        else:
            self.query_scale = MLP(d_model, d_model, d_model, 2, activation=nn.ReLU)
            self.ref_point_head = MLP(d_model, d_model, 2, 2, activation=nn.ReLU)

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        depth_pos_embed=None,
        mask_depth=None,
        bs=None,
        depth_pos_embed_ip=None,
        pos_embeds=None,
        attn_mask=None,
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_dims = []
        bs = src.shape[0]
        ###for dn
        if self.use_dab:
            reference_points = reference_points[None].repeat(bs, 1, 1)
        elif self.two_stgae_dino:
            reference_points = reference_points.sigmoid()

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
            if self.two_stgae_dino:
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
                raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256

                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            ###conditional
            # ipdb.set_trace()
            query_pos_un = None
            if self.use_dab:
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # bs, nq, 256*2
                raw_query_pos = self.ref_point_head(query_sine_embed)  # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                # pos_scale  = 1
                query_pos = pos_scale * raw_query_pos

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
                query_pos_un=query_pos_un,
            )

            # hack implementation for iterative bounding box refinement
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

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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
            "two_stage": False,
            "two_stage_num_proposals": 50,
            "use_dab": False,
            "two_stage_dino": False,
        },
    }

    def __new__(cls, model_name: str):
        return DepthAwareTransformer(**cls.CFG[model_name])
