# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTDETR decoder, modified from https://github.com/lyuwenyu/RT-DETR."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from typing import Any, Callable

import torch
import torchvision
from torch import nn
from torch.nn import init

from otx.algo.detection.utils.utils import (
    inverse_sigmoid,
)
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.transformer import deformable_attention_core_func

__all__ = ["RTDETRTransformer"]


def get_contrastive_denoising_training_group(
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    num_queries: int,
    class_embed: torch.nn.Module,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | tuple[None, None, None, None]:
    """Generate contrastive denoising training group.

    Args:
        targets (List[Dict[str, torch.Tensor]]): List of target dictionaries.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.nn.Module): Class embedding module.
        num_denoising (int, optional): Number of denoising queries. Defaults to 100.
        label_noise_ratio (float, optional): Ratio of label noise. Defaults to 0.5.
        box_noise_scale (float, optional): Scale of box noise. Defaults to 1.0.

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
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta


class MLP(nn.Module):
    """MLP."""

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
        value_spatial_shapes: list[tuple[int, int]],
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
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
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
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            msg = f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            raise ValueError(
                msg,
            )

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        return self.output_proj(output)


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer.

    Args:
        d_model (int): The number of expected features in the input.
        n_head (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.ReLU``.
        n_levels (int): The number of levels in MSDeformableAttention.
        n_points (int): The number of points in MSDeformableAttention.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """Initialize the TransformerDecoderLayer module."""
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Add positional embedding to the input tensor."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """Forward function of feed forward network."""
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: list[tuple[int, int]],
        memory_level_start_index: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        query_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward function of TransformerDecoderLayer."""
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)


class TransformerDecoder(nn.Module):
    """TransformerDecoder.

    Args:
        hidden_dim (int): The number of expected features in the input.
        decoder_layer (nn.Module): The decoder layer module.
        num_layers (int): The number of layers.
        eval_idx (int, optional): The index of evaluation layer.
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        tgt: torch.Tensor,
        ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: list[tuple[int, int]],
        memory_level_start_index: torch.Tensor,
        bbox_head: list[nn.Module],
        score_head: list[nn.Module],
        query_pos_head: nn.Module,
        attn_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = nn.functional.sigmoid(ref_points_unact)
        ref_points = ref_points_detach

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            inter_ref_bbox = nn.functional.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(nn.functional.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class RTDETRTransformer(BaseModule):
    """RTDETRTransformer.

    Args:
        num_classes (int): Number of object classes.
        hidden_dim (int): Hidden dimension size.
        num_queries (int): Number of queries.
        position_embed_type (str): Type of position embedding.
        feat_channels (List[int]): List of feature channels.
        feat_strides (List[int]): List of feature strides.
        num_levels (int): Number of levels.
        num_decoder_points (int): Number of decoder points.
        nhead (int): Number of attention heads.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.ReLU``.
        num_denoising (int): Number of denoising samples.
        label_noise_ratio (float): Ratio of label noise.
        box_noise_scale (float): Scale of box noise.
        learnt_init_query (bool): Whether to learn initial queries.
        eval_spatial_size (Tuple[int, int] | None): Spatial size for evaluation.
        eval_idx (int): Evaluation index.
        eps (float): Epsilon value.
        aux_loss (bool): Whether to include auxiliary loss.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        position_embed_type: str = "sine",
        feat_channels: list[int] = [512, 1024, 2048],  # noqa: B006
        feat_strides: list[int] = [8, 16, 32],  # noqa: B006
        num_levels: int = 3,
        num_decoder_points: int = 4,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        eval_spatial_size: tuple[int, int] | None = None,
        eval_idx: int = -1,
        eps: float = 1e-2,
        aux_loss: bool = True,
    ):
        """Initialize the RTDETRTransformer module."""
        super().__init__()
        if position_embed_type not in [
            "sine",
            "learned",
        ]:
            msg = f"position_embed_type not supported {position_embed_type}!"
            raise ValueError(msg)
        if len(feat_channels) > num_levels:
            msg = "Length of feat_channels should be less than or equal to num_levels."
            raise ValueError(msg)
        if len(feat_strides) != len(feat_channels):
            msg = "Length of feat_strides should be equal to length of feat_channels."
            raise ValueError(msg)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_decoder_points,
        )
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2, activation=activation)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3, activation=activation)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, num_layers=3, activation=activation) for _ in range(num_decoder_layers)],
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size is not None:
            self.anchors, self.valid_mask = self._generate_anchors()

    def init_weights(self) -> None:
        """Initialize the weights of the RTDETRTransformer."""
        prob = 0.01
        bias = float(-math.log((1 - prob) / prob))

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels: list[int]) -> None:
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ],
                    ),
                ),
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ],
                    ),
                ),
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: list[torch.Tensor]) -> tuple[Any, list[list[int]], list[int]]:
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(
        self,
        spatial_shapes: list[list[int]] | None = None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial_shapes is None:
            if self.eval_spatial_size is None:
                msg = "spatial_shapes or eval_spatial_size must be provided."
                raise ValueError(msg)
            anc_spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)] for s in self.feat_strides
            ]
        else:
            anc_spatial_shapes = spatial_shapes
        anchors = []
        for lvl, (h, w) in enumerate(anc_spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype),
                torch.arange(end=w, dtype=dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_wh = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        tensor_anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((tensor_anchors > self.eps) * (tensor_anchors < 1 - self.eps)).all(-1, keepdim=True)
        tensor_anchors = torch.log(tensor_anchors / (1 - tensor_anchors))
        tensor_anchors = torch.where(valid_mask, tensor_anchors, torch.inf)

        return tensor_anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: list[list[int]],
        denoising_class: torch.Tensor | None = None,
        denoising_bbox_unact: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]),
        )

        enc_topk_bboxes = nn.functional.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]),
        )

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, feats: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None) -> torch.Tensor:
        """Forward pass of the RTDETRTransformer module."""
        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_class,
            denoising_bbox_unact,
        )

        # decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out["aux_outputs"].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: torch.Tensor, outputs_coord: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]
