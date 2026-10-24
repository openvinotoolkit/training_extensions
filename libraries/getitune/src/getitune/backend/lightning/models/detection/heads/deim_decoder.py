# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DEIM Transformer Decoder.

Modified from DEIMv2 (https://github.com/Intellindust-AI-Lab/DEIMv2)
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.nn import init

from getitune.backend.lightning.models.common.layers.transformer_layers import (
    LQE,
    MLP,
    Gate,
    Integral,
    MSDeformableAttentionV2,
    SwiGLUFFN,
    get_contrastive_denoising_training_group,
)
from getitune.backend.lightning.models.common.utils.utils import inverse_sigmoid
from getitune.backend.lightning.models.detection.utils.utils import dfine_distance2bbox, dfine_weighting_function
from getitune.backend.lightning.models.modules.norm import RMSNorm
from getitune.backend.lightning.models.utils.weight_init import bias_init_with_prob

if TYPE_CHECKING:
    from torch.nn import ModuleList

__all__ = ["DEIMTransformer"]


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention, cross-attention, and FFN.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        n_levels: Number of feature levels for deformable attention.
        n_points: Number of sampling points per level.
        layer_scale: Optional scale factor for wide layers.
        use_gateway: Whether to use gated fusion for cross-attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        n_levels: int = 4,
        n_points: int | list[int] = 4,
        layer_scale: float | None = None,
        use_gateway: bool = False,
    ) -> None:
        super().__init__()

        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention - use memory-efficient scaled_dot_product_attention
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        # cross attention
        n_points_list = [n_points] * n_levels if isinstance(n_points, int) else n_points
        self.cross_attn = MSDeformableAttentionV2(d_model, n_head, n_levels, n_points_list)
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        # ffn
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        """Add positional embedding to tensor if provided."""
        return tensor if pos is None else tensor + pos

    def _self_attention(self, qk: Tensor, v: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Self-attention with fused Q/K projection and a separate V projection.

        Q and K are projected together from the position-embedded queries, while V is
        projected from the raw target (matching ``nn.MultiheadAttention(q, k, value=tgt)``
        semantics used by upstream DEIMv2).

        Args:
            qk: Position-embedded query features used for both Q and K, shape (B, N, C).
            v: Raw target features used for V (no positional embedding), shape (B, N, C).
            attn_mask: Optional attention mask of shape (N, N) or (B, N, N).

        Returns:
            Attention output of shape (B, N, C).
        """
        B, N, C = qk.shape  # noqa: N806
        weight, bias = self.qkv_proj.weight, self.qkv_proj.bias

        # Fused Q/K projection (one matmul), separate V projection from the raw target.
        qk_proj = f.linear(qk, weight[: 2 * C], bias[: 2 * C])
        q, k = qk_proj.reshape(B, N, 2, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        v = f.linear(v, weight[2 * C :], bias[2 * C :]).reshape(B, N, self.n_head, self.head_dim).transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        out = f.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, C))

    def forward(
        self,
        target: Tensor,
        reference_points: Tensor,
        value: tuple[Tensor, ...],
        spatial_shapes: list[list[int]],
        attn_mask: Tensor | None = None,
        query_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through decoder layer.

        Args:
            target: Query features of shape (B, N, C).
            reference_points: Reference points of shape (B, N, 1, 4).
            value: Multi-scale value features.
            spatial_shapes: Spatial shapes of each feature level.
            attn_mask: Optional attention mask.
            query_pos_embed: Optional positional embedding for queries.

        Returns:
            Updated query features of shape (B, N, C).
        """
        # self attention: Q/K use position-embedded queries, V uses the raw target
        qk = self.with_pos_embed(target, query_pos_embed)
        target2 = self._self_attention(qk, target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes)

        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)

        # ffn
        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        return self.norm3(target.clamp(min=-65504, max=65504))


class TransformerDecoder(nn.Module):
    """Transformer Decoder with Fine-grained Distribution Refinement (FDR).

    Refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement
    techniques to improve bounding box accuracy.

    Args:
        hidden_dim: Hidden dimension.
        decoder_layer: Standard decoder layer.
        decoder_layer_wide: Wide decoder layer for later stages.
        num_layers: Total number of decoder layers.
        num_head: Number of attention heads.
        reg_max: Maximum regression bins.
        reg_scale: Regression scale factor.
        up: Up-sampling parameter.
        eval_idx: Index of layer used for evaluation.
        layer_scale: Scale factor for wide layers.
        act: Activation function class.
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: TransformerDecoderLayer,
        decoder_layer_wide: TransformerDecoderLayer,
        num_layers: int,
        num_head: int,
        reg_max: int,
        reg_scale: nn.Parameter,
        up: nn.Parameter,
        eval_idx: int = -1,
        layer_scale: int = 2,
        act: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max, activation=act)) for _ in range(num_layers)]
        )

    def value_op(
        self,
        memory: Tensor,
        value_proj: nn.Module | None,
        value_scale: int | None,
        memory_mask: Tensor | None,
        memory_spatial_shapes: list[list[int]],
    ) -> tuple[Tensor, ...]:
        """Preprocess values for MSDeformableAttention.

        Args:
            memory: Encoder memory of shape (B, L, C).
            value_proj: Optional projection layer.
            value_scale: Optional scale for interpolation.
            memory_mask: Optional memory mask.
            memory_spatial_shapes: Spatial shapes of each level.

        Returns:
            Tuple of value tensors split by level.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = f.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self) -> None:
        """Convert model for deployment by removing unused layers."""
        self.project = dfine_weighting_function(self.reg_max, self.up, self.reg_scale)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]])

    def forward(
        self,
        target: Tensor,
        ref_points_unact: Tensor,
        memory: Tensor,
        spatial_shapes: list[list[int]],
        bbox_head: ModuleList,
        score_head: ModuleList,
        query_pos_head: MLP,
        pre_bbox_head: MLP,
        integral: Integral,
        up: nn.Parameter,
        reg_scale: nn.Parameter,
        attn_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass through decoder.

        Args:
            target: Query features of shape (B, N, C).
            ref_points_unact: Unactivated reference points of shape (B, N, 4).
            memory: Encoder memory of shape (B, L, C).
            spatial_shapes: Spatial shapes of each feature level.
            bbox_head: Bounding box regression heads.
            score_head: Classification heads.
            query_pos_head: Query position embedding head.
            pre_bbox_head: Pre-bbox head for initial predictions.
            integral: Integral layer for distribution regression.
            up: Up-sampling parameter.
            reg_scale: Regression scale parameter.
            attn_mask: Optional attention mask.
            memory_mask: Optional memory mask.
            dn_meta: Optional denoising metadata.

        Returns:
            Tuple of (bboxes, logits, corners, refs, pre_bboxes, pre_scores).
        """
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(self, "project"):
            project = dfine_weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = f.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = f.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = f.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = f.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = dfine_distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


class DEIMTransformerModule(nn.Module):
    """DEIM Transformer module for object detection.

    This module implements the DEIM (Detection Transformer with Efficient
    Integration Module) architecture with Fine-grained Distribution Refinement
    (FDR) for accurate object detection.

    Attributes:
        __share__: List of attributes shared across instances.
        hidden_dim: Hidden dimension size.
        nhead: Number of attention heads.
        feat_strides: Feature strides for each level.
        num_levels: Number of feature levels.
        num_classes: Number of object classes.
        num_queries: Number of detection queries.
        eps: Small epsilon for numerical stability.
        num_layers: Number of decoder layers.
        eval_spatial_size: Spatial size for evaluation.
        aux_loss: Whether to use auxiliary losses.
        reg_max: Maximum regression value for FDR.
    """

    __share__: ClassVar[list[str]] = ["num_classes", "eval_spatial_size"]

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: list[int] | None = None,
        feat_strides: list[int] | None = None,
        num_levels: int = 3,
        num_points: list[int] | None = None,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.SiLU,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learn_query_content: bool = False,
        eval_spatial_size: tuple[int, int] | None = None,
        eval_idx: int = -1,
        eps: float = 1e-2,
        aux_loss: bool = True,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        reg_max: int = 32,
        reg_scale: float = 4.0,
        layer_scale: int = 1,
        use_gateway: bool = True,
        share_bbox_head: bool = False,
        share_score_head: bool = False,
    ) -> None:
        """Initialize DEIMTransformerModule.

        Args:
            num_classes: Number of object classes.
            hidden_dim: Hidden dimension size.
            num_queries: Number of detection queries.
            feat_channels: Feature channels for each input level.
            feat_strides: Feature strides for each level.
            num_levels: Number of feature levels.
            num_points: Number of sampling points per level.
            nhead: Number of attention heads.
            num_layers: Number of decoder layers.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout rate.
            activation: Activation function class.
            num_denoising: Number of denoising queries for training.
            label_noise_ratio: Label noise ratio for denoising.
            box_noise_scale: Box noise scale for denoising.
            learn_query_content: Whether to learn query content.
            eval_spatial_size: Spatial size for evaluation (H, W).
            eval_idx: Evaluation layer index (-1 for last).
            eps: Epsilon for numerical stability.
            aux_loss: Whether to use auxiliary losses.
            cross_attn_method: Cross attention method ('default' or 'discrete').
            query_select_method: Query selection method.
            reg_max: Maximum regression value for FDR.
            reg_scale: Regression scale factor.
            layer_scale: Scale factor for wide layers.
            use_gateway: Whether to use gateway fusion.
            share_bbox_head: Whether to share bbox head across layers.
            share_score_head: Whether to share score head across layers.
        """
        super().__init__()
        if feat_channels is None:
            feat_channels = [256, 256, 256]
        if feat_strides is None:
            feat_strides = [8, 16, 32]
        if num_points is None:
            num_points = [3, 6, 3]
        if len(feat_channels) > num_levels:
            msg = f"feat_channels ({len(feat_channels)}) must be <= num_levels ({num_levels})"
            raise ValueError(msg)
        if len(feat_strides) != len(feat_channels):
            msg = f"feat_strides ({len(feat_strides)}) must match feat_channels ({len(feat_channels)})"
            raise ValueError(msg)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            num_levels,
            num_points,
            use_gateway=use_gateway,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            num_levels,
            num_points,
            layer_scale=layer_scale,
            use_gateway=use_gateway,
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=partial(activation, inplace=True),
        )
        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, activation=partial(activation, inplace=True))

        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, activation=partial(activation, inplace=True))

        # decoder head
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, activation=partial(activation, inplace=True))
        self.integral = Integral(self.reg_max)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        dec_score_head = nn.Linear(hidden_dim, num_classes)
        self.dec_score_head = nn.ModuleList(
            [dec_score_head if share_score_head else copy.deepcopy(dec_score_head) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(dec_score_head) for _ in range(num_layers - self.eval_idx - 1)]
        )

        # Share the same bbox head for all layers
        dec_bbox_head = MLP(
            hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, activation=partial(activation, inplace=True)
        )
        self.dec_bbox_head = nn.ModuleList(
            [dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head) for _ in range(self.eval_idx + 1)]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, activation=partial(activation, inplace=True))
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self) -> None:
        """Convert model to deployment mode by pruning unused components."""
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels: list[int]) -> None:
        """Reset model parameters with appropriate initialization.

        Args:
            feat_channels: List of feature channel dimensions.
        """
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels: list[int]) -> None:
        """Build input projection layers for feature transformation.

        Args:
            feat_channels: List of input feature channel dimensions.
        """
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: list[Tensor]) -> tuple[Tensor, list[list[int]]]:
        """Get encoder input from multi-scale features.

        Projects input features to hidden dimension and flattens them
        for transformer processing.

        Args:
            feats: List of feature tensors from backbone.

        Returns:
            Tuple of (flattened features, spatial shapes per level).
        """
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
        for feat in proj_feats:
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes: list[list[int]] | None = None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Generate anchor points for all feature levels.

        Args:
            spatial_shapes: Spatial shapes for each level. If None, computed from eval_spatial_size.
            grid_size: Base grid size for anchors.
            dtype: Data type for anchor tensors.
            device: Device to place anchor tensors on.

        Returns:
            Tuple of (anchor coordinates, validity mask).
        """
        if spatial_shapes is None:
            if self.eval_spatial_size is None:
                msg = "eval_spatial_size must be set when spatial_shapes is None"
                raise ValueError(msg)
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchor_list: list[Tensor] = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchor_list.append(lvl_anchors)

        anchors = torch.concat(anchor_list, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: Tensor,
        spatial_shapes: list[list[int]],
        denoising_logits: Tensor | None = None,
        denoising_bbox_unact: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor], Tensor]:
        """Prepare input for the decoder.

        Generates anchors, selects top-k queries, and prepares content
        embeddings for decoder processing.

        Args:
            memory: Encoder memory of shape (B, L, C).
            spatial_shapes: Spatial shapes for each feature level.
            denoising_logits: Optional denoising logits for training.
            denoising_bbox_unact: Optional denoising bbox for training.

        Returns:
            Tuple of (content, bbox_unact, topk_bboxes_list, topk_logits_list, enc_logits).
        """
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory

        enc_outputs_logits: Tensor = self.enc_score_head(memory)

        # select topk queries
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            memory, enc_outputs_logits, anchors, self.num_queries
        )

        enc_topk_bbox_unact: Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = f.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            if denoising_logits is not None:
                content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits

    def _select_topk(
        self,
        memory: Tensor,
        outputs_logits: Tensor,
        outputs_anchors_unact: Tensor,
        topk: int,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Select top-k queries based on classification scores.

        Args:
            memory: Encoder memory of shape (B, L, C).
            outputs_logits: Classification logits of shape (B, L, num_classes).
            outputs_anchors_unact: Unactivated anchor coordinates.
            topk: Number of top queries to select.

        Returns:
            Tuple of (topk_memory, topk_logits, topk_anchors).
        """
        topk_ind: Tensor
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_anchors = outputs_anchors_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1])
        )

        topk_logits = (
            outputs_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
            if self.training
            else None
        )

        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors

    def forward(
        self,
        feats: list[Tensor],
        targets: list[dict[str, Any]] | None = None,
        explain_mode: bool = False,
    ) -> dict[str, Any]:
        """Forward pass of the DEIM Transformer module.

        Args:
            feats: List of multi-scale feature tensors from backbone.
            targets: Optional list of target dictionaries for training.
            explain_mode: Whether to include raw logits for explainability.

        Returns:
            Dictionary containing predictions and optional auxiliary outputs:
                - pred_logits: Classification logits.
                - pred_boxes: Predicted bounding boxes.
                - pred_corners: Corner predictions (training only).
                - ref_points: Reference points (training only).
                - aux_outputs: Auxiliary outputs from intermediate layers.
                - dn_outputs: Denoising outputs (training only).
        """
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=1.0,
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits = (
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
        )

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
        )

        out_bboxes = out_bboxes.clamp(min=1e-8)

        if self.training and dn_meta is not None:
            # the output from the first decoder layer, only one
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)

            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)

            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1], out_corners[-1], out_logits[-1]
            )
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs, dn_out_corners[-1], dn_out_logits[-1]
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

        if explain_mode:
            out["raw_logits"] = enc_outputs_logits

        return out

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class: list[Tensor],
        outputs_coord: list[Tensor],
    ) -> list[dict[str, Tensor]]:
        """Set auxiliary loss outputs for encoder.

        This is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values.

        Args:
            outputs_class: List of classification outputs.
            outputs_coord: List of coordinate outputs.

        Returns:
            List of dictionaries with pred_logits and pred_boxes.
        """
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class: list[Tensor],
        outputs_coord: list[Tensor],
        outputs_corners: list[Tensor],
        outputs_ref: list[Tensor],
        teacher_corners: Tensor | None = None,
        teacher_logits: Tensor | None = None,
    ) -> list[dict[str, Tensor | None]]:
        """Set auxiliary loss outputs for decoder with FDR.

        This is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values.

        Args:
            outputs_class: List of classification outputs.
            outputs_coord: List of coordinate outputs.
            outputs_corners: List of corner outputs.
            outputs_ref: List of reference point outputs.
            teacher_corners: Optional teacher corner predictions.
            teacher_logits: Optional teacher logits.

        Returns:
            List of dictionaries with predictions and teacher outputs.
        """
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
        ]


class DEIMTransformer:
    """Factory class for creating DEIMTransformerModule instances.

    Provides predefined configurations for different model sizes (x, l, m, s)
    with appropriate hidden dimensions, number of layers, and feedforward dimensions.

    Attributes:
        decoder_cfg: Dictionary mapping model names to their configurations.
    """

    decoder_cfg: ClassVar[dict[str, Any]] = {
        "deimv2_x": {
            "num_layers": 6,
            "eval_idx": -1,
            "feat_channels": [256, 256, 256],
            "hidden_dim": 256,
            "dim_feedforward": 2048,
        },
        "deimv2_l": {
            "feat_channels": [224, 224, 224],
            "hidden_dim": 224,
            "num_layers": 4,
            "eval_idx": -1,
            "dim_feedforward": 1792,
        },
        "deimv2_m": {
            "feat_channels": [256, 256, 256],
            "hidden_dim": 256,
            "dim_feedforward": 512,
            "num_layers": 4,
            "eval_idx": -1,
        },
        "deimv2_s": {
            "feat_channels": [192, 192, 192],
            "hidden_dim": 192,
            "dim_feedforward": 512,
            "num_layers": 4,
            "eval_idx": -1,
        },
    }

    def __new__(
        cls,
        model_name: str,
        num_classes: int,
        eval_spatial_size: tuple[int, int] = (640, 640),
    ) -> DEIMTransformerModule:
        """Create a new DEIMTransformerModule instance.

        Args:
            model_name: Name of the model configuration (e.g., 'deimv2_x').
            num_classes: Number of object classes.
            eval_spatial_size: Spatial size for evaluation (H, W).

        Returns:
            Configured DEIMTransformerModule instance.

        Raises:
            KeyError: If model_name is not found in decoder_cfg.
        """
        cfg = cls.decoder_cfg[model_name]
        return DEIMTransformerModule(num_classes=num_classes, eval_spatial_size=eval_spatial_size, **cfg)
