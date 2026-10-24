# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EdgeCrafter Transformer Decoder (ECTransformer).

Modified from EdgeCrafter (https://github.com/Intellindust-AI-Lab/EdgeCrafter).
Modified from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR).
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn import init

from getitune.backend.lightning.models.common.layers.transformer_layers import (
    LQE,
    MLP,
    Gate,
    Integral,
    MSDeformableAttentionV2,
    get_contrastive_denoising_training_group,
)
from getitune.backend.lightning.models.common.utils.utils import inverse_sigmoid
from getitune.backend.lightning.models.detection.utils.utils import dfine_distance2bbox, dfine_weighting_function
from getitune.backend.lightning.models.utils.weight_init import bias_init_with_prob

if TYPE_CHECKING:
    # Deferred: importing the *package* `instance_segmentation.heads` eagerly executes
    # `instance_segmentation/__init__.py`, which re-exports `EdgeCrafterInst`, which in
    # turn imports `EdgeCrafterMixin` -> this module, completing a circular import.
    # Safe here because `from __future__ import annotations` makes every annotation in
    # this file lazy; the real class is imported locally where it is instantiated below.
    from getitune.backend.lightning.models.instance_segmentation.heads import SegmentationHead

__all__ = ["ECTransformer"]

# ---------------------------------------------------------------------------
# Per-model hyper-parameters
# ---------------------------------------------------------------------------

_MODEL_CFGS: dict[str, dict[str, Any]] = {
    "edgecrafter_s": {
        "hidden_dim": 192,
        "num_heads": 8,
        "dim_feedforward": 512,
        "num_layers": 4,
        "feat_channels": [192, 192, 192],
    },
    "edgecrafter_m": {
        "hidden_dim": 256,
        "num_heads": 8,
        "dim_feedforward": 1024,
        "num_layers": 4,
        "feat_channels": [256, 256, 256],
    },
    "edgecrafter_l": {
        "hidden_dim": 256,
        "num_heads": 8,
        "dim_feedforward": 1024,
        "num_layers": 4,
        "feat_channels": [256, 256, 256],
    },
    "edgecrafter_x": {
        "hidden_dim": 256,
        "num_heads": 8,
        "dim_feedforward": 2048,
        "num_layers": 4,
        "feat_channels": [256, 256, 256],
    },
}

# ---------------------------------------------------------------------------
# Transformer Decoder Layer
# ---------------------------------------------------------------------------


class ECTransformerDecoderLayer(nn.Module):
    """Single EC-Transformer decoder layer.

    Uses:
    * ``nn.MultiheadAttention`` for self-attention (matches checkpoint keys).
    * ``MSDeformableAttentionV2`` for cross-attention.
    * ``Gate`` for gated cross-attention fusion.
    * Standard Linear FFN with SiLU activation.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        n_levels: Number of feature levels.
        n_points: Number of deformable sampling points per level (int or list).
        layer_scale: Optional scale factor for enlarged wide layers.
        activation: Activation layer class.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        n_levels: int = 3,
        n_points: int | list[int] = 4,
        layer_scale: float | None = None,
        activation: Callable[..., nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self-attention (uses nn.MultiheadAttention to match checkpoint key layout)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross-attention
        n_points_list = [n_points] * n_levels if isinstance(n_points, int) else list(n_points)
        self.cross_attn = MSDeformableAttentionV2(d_model, n_head, n_levels, n_points_list)
        self.dropout2 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def _with_pos(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        target: Tensor,
        reference_points: Tensor,
        value: list[Tensor],
        spatial_shapes: list[list[int]],
        attn_mask: Tensor | None = None,
        query_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """Forward pass."""
        # self-attention
        q = k = self._with_pos(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = self.norm1(target + self.dropout1(target2))

        # cross-attention
        target2 = self.cross_attn(
            self._with_pos(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )
        target = self.gateway(target, self.dropout2(target2))

        # FFN
        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        return self.norm2((target + self.dropout4(target2)).clamp(min=-65504, max=65504))


# ---------------------------------------------------------------------------
# Transformer Decoder (iterative FDR)
# ---------------------------------------------------------------------------


class ECTransformerDecoder(nn.Module):
    """Iterative Fine-grained Distribution Refinement (FDR) decoder."""

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: ECTransformerDecoderLayer,
        decoder_layer_wide: ECTransformerDecoderLayer,
        segmentation_head: SegmentationHead | None,
        num_layers: int,
        num_head: int,
        reg_max: int,
        reg_scale: Tensor,
        up: Tensor,
        eval_idx: int = -1,
        layer_scale: int = 1,
        act: Callable[..., nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up = up
        self.reg_scale = reg_scale
        self.reg_max = reg_max

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.segmentation_head = segmentation_head
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max, activation=act)) for _ in range(num_layers)],
        )

    def _value_op(
        self,
        memory: Tensor,
        memory_spatial_shapes: list[list[int]],
    ) -> list[Tensor]:
        """Pre-process encoder memory into per-level value tensors."""
        bs = memory.shape[0]
        memory_reshaped = memory.reshape(bs, memory.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return list(memory_reshaped.permute(0, 2, 3, 1).split(split_shape, dim=-1))

    def convert_to_deploy(self) -> None:
        """Trim layers and freeze LQE for deployment."""
        project = dfine_weighting_function(self.reg_max, self.up, self.reg_scale)
        self.register_buffer("project", project)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]]  # type: ignore[list-item]
        )

    def forward(
        self,
        spatial_feat: Tensor | None,
        target: Tensor,
        ref_points_unact: Tensor,
        memory: Tensor,
        spatial_shapes: list[list[int]],
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        query_pos_head: nn.Module,
        pre_bbox_head: nn.Module,
        integral: Integral,
        up: Tensor,
        reg_scale: Tensor,
        attn_mask: Tensor | None = None,
        dn_meta: dict | None = None,
    ) -> tuple:
        """Iterative decode pass."""
        output = target
        output_detach = pred_corners_undetach = 0

        value = self._value_op(memory, spatial_shapes)

        dec_out_bboxes: list[Tensor] = []
        dec_out_logits: list[Tensor] = []
        dec_out_pred_corners: list[Tensor] = []
        dec_out_refs: list[Tensor] = []
        dec_out_hs: list[Tensor] = []

        if not hasattr(self, "project"):
            project = dfine_weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project  # type: ignore[attr-defined]

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        ref_points_initial = ref_points_detach
        pre_bboxes = ref_points_detach
        pre_scores: Tensor = ref_points_detach

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=float(self.layer_scale))
                value = self._value_op(memory, spatial_shapes)  # recompute for new dim
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach  # type: ignore[operator]
            inter_ref_bbox = dfine_distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = self.lqe_layers[i](score_head[i](output), pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                dec_out_hs.append(output)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        # Segmentation
        if spatial_feat is not None and self.segmentation_head is not None:
            dec_out_segs = self.segmentation_head(
                spatial_features=spatial_feat,
                query_features=dec_out_hs,
            )
            return (
                torch.stack(dec_out_bboxes),
                torch.stack(dec_out_logits),
                torch.stack(dec_out_pred_corners),
                torch.stack(dec_out_refs),
                torch.stack(dec_out_segs),
                pre_bboxes,
                pre_scores,
                dec_out_segs[-1],
            )

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            None,
            pre_bboxes,
            pre_scores,
            None,
        )


# ---------------------------------------------------------------------------
# ECTransformer (top-level decoder module)
# ---------------------------------------------------------------------------


class ECTransformer(nn.Module):
    """EdgeCrafter Transformer decoder.

    Handles both detection (``spatial_feat=None``) and instance segmentation
    (``spatial_feat`` is the stride-8 backbone feature map).

    Args:
        model_name: One of ``edgecrafter_{s,m,l,x}``.
        num_classes: Number of object classes.
        eval_spatial_size: ``(H, W)`` used to pre-compute static anchor cache.
        num_queries: Number of object queries.
        num_levels: Number of encoder feature levels.
        num_points: Deformable attention points per level (int or list).
        dropout: Dropout rate.
        activation: FFN activation layer class.
        num_denoising: Number of denoising groups.
        label_noise_ratio: Label noise ratio for CDN.
        box_noise_scale: Box noise scale for CDN.
        reg_max: Bin count for distribution-based box regression.
        reg_scale: Curvature parameter for weighting function.
        layer_scale: Dimension scale factor for wide decoder layers (usually 1).
        mask_downsample_ratio: When set, creates a ``SegmentationHead`` with this
            output/input stride ratio (e.g. 4 for 160x160 masks from 640 input).
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 80,
        eval_spatial_size: tuple[int, int] | list[int] = (640, 640),
        num_queries: int = 300,
        num_levels: int = 3,
        num_points: int | list[int] = [3, 6, 3],  # noqa: B006
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.SiLU,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        reg_max: int = 32,
        reg_scale: float = 4.0,
        layer_scale: int = 1,
        mask_downsample_ratio: int | None = None,
    ) -> None:
        super().__init__()
        if model_name not in _MODEL_CFGS:
            msg = f"Unknown EdgeCrafter model name '{model_name}'. Available: {list(_MODEL_CFGS)}"
            raise ValueError(msg)

        cfg = _MODEL_CFGS[model_name]
        hidden_dim: int = cfg["hidden_dim"]
        nhead: int = cfg["num_heads"]
        dim_feedforward: int = cfg["dim_feedforward"]
        num_layers: int = cfg["num_layers"]
        feat_channels: list[int] = cfg["feat_channels"]

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.eps = 1e-2
        self.aux_loss = True
        self.reg_max = reg_max
        self.eval_spatial_size = tuple(eval_spatial_size)
        self.feat_strides = [8, 16, 32]

        # Learnable reg params (frozen after init)
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        # Input projections
        self._build_input_proj_layer(feat_channels)

        eval_idx = -1  # always use last layer at eval

        # Decoder layers
        n_points_list = [num_points] * num_levels if isinstance(num_points, int) else list(num_points)
        decoder_layer = ECTransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, num_levels, n_points_list, activation=activation
        )
        decoder_layer_wide = ECTransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            num_levels,
            n_points_list,
            layer_scale=float(layer_scale) if layer_scale > 1 else None,
            activation=activation,
        )

        seg_head: SegmentationHead | None = None
        if mask_downsample_ratio is not None:
            from getitune.backend.lightning.models.instance_segmentation.heads import (
                SegmentationHead as _SegmentationHead,
            )

            seg_head = _SegmentationHead(
                in_dim=hidden_dim,
                num_blocks=num_layers,
                downsample_ratio=mask_downsample_ratio,
                image_size=cast("tuple[int, int]", self.eval_spatial_size),
            )

        self.decoder = ECTransformerDecoder(
            hidden_dim=hidden_dim,
            decoder_layer=decoder_layer,
            decoder_layer_wide=decoder_layer_wide,
            segmentation_head=seg_head,
            num_layers=num_layers,
            num_head=nhead,
            reg_max=reg_max,
            reg_scale=self.reg_scale,
            up=self.up,
            eval_idx=eval_idx,
            layer_scale=layer_scale,
            act=activation,
        )

        # Denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # Encoder output heads
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, activation=activation)
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, activation=activation)
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, activation=activation)
        self.integral = Integral(reg_max)

        actual_eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self._eval_idx = actual_eval_idx

        dec_score_head = nn.Linear(hidden_dim, num_classes)
        self.dec_score_head = nn.ModuleList(
            [copy.deepcopy(dec_score_head) for _ in range(actual_eval_idx + 1)]
            + [copy.deepcopy(dec_score_head) for _ in range(num_layers - actual_eval_idx - 1)]
        )

        scaled_dim = round(layer_scale * hidden_dim)
        dec_bbox_head = MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3, activation=activation)
        wide_bbox_head = MLP(scaled_dim, scaled_dim, 4 * (reg_max + 1), 3, activation=activation)
        self.dec_bbox_head = nn.ModuleList(
            [copy.deepcopy(dec_bbox_head) for _ in range(actual_eval_idx + 1)]
            + [
                wide_bbox_head if i > actual_eval_idx else copy.deepcopy(dec_bbox_head)
                for i in range(actual_eval_idx + 1, num_layers)
            ]
        )

        # Static anchor cache
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        self._reset_parameters(feat_channels)

    # ------------------------------------------------------------------
    def _build_input_proj_layer(self, feat_channels: list[int]) -> None:
        self.input_proj = nn.ModuleList()
        for in_ch in feat_channels:
            if in_ch == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                _layers1: list[tuple[str, nn.Module]] = [
                    ("conv", nn.Conv2d(in_ch, self.hidden_dim, 1, bias=False)),
                    ("norm", nn.BatchNorm2d(self.hidden_dim)),
                ]
                self.input_proj.append(nn.Sequential(OrderedDict(_layers1)))
        for _ in range(self.num_levels - len(feat_channels)):
            in_ch = feat_channels[-1]
            _layers2: list[tuple[str, nn.Module]] = [
                ("conv", nn.Conv2d(in_ch, self.hidden_dim, 3, 2, padding=1, bias=False)),
                ("norm", nn.BatchNorm2d(self.hidden_dim)),
            ]
            self.input_proj.append(nn.Sequential(OrderedDict(_layers2)))

    def _reset_parameters(self, feat_channels: list[int]) -> None:
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        _enc_last = cast("nn.Linear", self.enc_bbox_head.layers[-1])
        init.constant_(_enc_last.weight, 0)
        init.constant_(_enc_last.bias, 0)
        _pre_last = cast("nn.Linear", self.pre_bbox_head.layers[-1])
        init.constant_(_pre_last.weight, 0)
        init.constant_(_pre_last.bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cast("nn.Linear", cls_).bias, bias)
            if hasattr(reg_, "layers"):
                _reg = cast("MLP", reg_)
                _reg_last = cast("nn.Linear", _reg.layers[-1])
                init.constant_(_reg_last.weight, 0)
                init.constant_(_reg_last.bias, 0)
        _q0 = cast("nn.Linear", self.query_pos_head.layers[0])
        init.xavier_uniform_(_q0.weight)
        _q1 = cast("nn.Linear", self.query_pos_head.layers[1])
        init.xavier_uniform_(_q1.weight)
        _qlast = cast("nn.Linear", self.query_pos_head.layers[-1])
        init.xavier_uniform_(_qlast.weight)

    def _generate_anchors(
        self,
        spatial_shapes: list[list[int]] | None = None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        if spatial_shapes is None:
            spatial_shapes = [
                [self.eval_spatial_size[0] // s, self.eval_spatial_size[1] // s] for s in self.feat_strides
            ]
        anchors: list[Tensor] = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([gx, gy], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.cat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)
        anchors_cat = torch.cat(anchors, dim=1).to(device)
        valid_mask = ((anchors_cat > self.eps) & (anchors_cat < 1 - self.eps)).all(-1, keepdim=True)
        anchors_cat = torch.log(anchors_cat / (1 - anchors_cat))
        anchors_cat = torch.where(valid_mask, anchors_cat, torch.full_like(anchors_cat, float("inf")))
        return anchors_cat, valid_mask

    def _get_encoder_input(self, feats: list[Tensor]) -> tuple[Tensor, list[list[int]]]:
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        feat_flatten: list[Tensor] = []
        spatial_shapes: list[list[int]] = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        return torch.cat(feat_flatten, 1), spatial_shapes

    def _select_topk(
        self,
        memory: Tensor,
        outputs_logits: Tensor,
        outputs_anchors_unact: Tensor,
        topk: int,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).expand(-1, -1, outputs_anchors_unact.shape[-1]),
        )
        topk_logits = (
            outputs_logits.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).expand(-1, -1, outputs_logits.shape[-1]),
            )
            if self.training
            else None
        )
        topk_memory = memory.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).expand(-1, -1, memory.shape[-1]),
        )
        return topk_memory, topk_logits, topk_anchors

    def _get_decoder_input(
        self,
        memory: Tensor,
        spatial_shapes: list[list[int]],
        denoising_logits: Tensor | None = None,
        denoising_bbox_unact: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor | None]]:
        if self.training or not hasattr(self, "anchors"):
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors: Tensor = self.anchors  # type: ignore[attr-defined]
            valid_mask: Tensor = self.valid_mask  # type: ignore[attr-defined]

        if memory.shape[0] > 1:
            anchors = anchors.expand(memory.shape[0], -1, -1)

        memory = valid_mask.to(memory.dtype) * memory
        enc_outputs_logits = self.enc_score_head(memory)

        topk_memory, topk_logits, topk_anchors = self._select_topk(
            memory, enc_outputs_logits, anchors, self.num_queries
        )
        enc_topk_bbox_unact = self.enc_bbox_head(topk_memory) + topk_anchors

        enc_bboxes_list: list[Tensor] = []
        enc_logits_list: list[Tensor | None] = []
        if self.training:
            enc_bboxes_list.append(F.sigmoid(enc_topk_bbox_unact))
            enc_logits_list.append(topk_logits)

        content = topk_memory.detach()
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None and denoising_logits is not None:
            enc_topk_bbox_unact = torch.cat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.cat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_bboxes_list, enc_logits_list

    @staticmethod
    def _split(x: Tensor | None, dim: int, s_idx: Tensor | list[int] | None) -> tuple[Tensor | None, Tensor | None]:
        if x is None or s_idx is None:
            return None, x
        split_sizes: list[int] = s_idx.tolist() if isinstance(s_idx, Tensor) else s_idx
        return torch.split(x, split_sizes, dim=dim)  # type: ignore[return-value]

    def forward(
        self,
        feats: list[Tensor],
        targets: list[dict] | None = None,
        spatial_feat: Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            feats: List of multi-scale features from the encoder.
            targets: Ground-truth targets (required during training).
            spatial_feat: Stride-8 spatial feature for segmentation head (ECSeg only).

        Returns:
            Dict with ``pred_logits``, ``pred_boxes``, optional ``pred_masks``,
            and auxiliary losses during training.
        """
        memory, spatial_shapes = self._get_encoder_input(feats)

        # Denoising
        denoising_logits = denoising_bbox_unact = attn_mask = dn_meta = None
        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        out_bboxes, out_logits, out_corners, out_refs, out_masks, pre_bboxes, pre_logits, pre_segs = self.decoder(
            spatial_feat,
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

        s_idx = dn_meta["dn_num_split"] if dn_meta is not None else None

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = self._split(pre_logits, 1, s_idx)
            dn_pre_bboxes, pre_bboxes = self._split(pre_bboxes, 1, s_idx)
            dn_pre_segs, pred_segs = self._split(pre_segs, 1, s_idx)
            dn_out_logits, out_logits = self._split(out_logits, 2, s_idx)
            dn_out_bboxes, out_bboxes = self._split(out_bboxes, 2, s_idx)
            dn_out_masks, out_masks = self._split(out_masks, 2, s_idx)
            dn_out_corners, out_corners = self._split(out_corners, 2, s_idx)
            dn_out_refs, out_refs = self._split(out_refs, 2, s_idx)
        else:
            pred_segs = pre_segs
            dn_out_logits = dn_out_bboxes = dn_out_masks = dn_out_corners = dn_out_refs = None
            dn_pre_logits = dn_pre_bboxes = dn_pre_segs = None

        last_masks = out_masks[-1] if out_masks is not None else None
        if self.training:
            out: dict[str, Any] = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
            if last_masks is not None:
                out["pred_masks"] = last_masks
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
            }
            if last_masks is not None:
                out["pred_masks"] = last_masks

        if self.training and self.aux_loss:
            aux_masks: list[Tensor | None] = (
                list(out_masks[:-1]) if out_masks is not None else [None] * (len(out_logits) - 1)
            )
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                aux_masks,
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            pre_out: dict[str, Any] = {
                "pred_logits": pre_logits,
                "pred_boxes": pre_bboxes,
            }
            if pred_segs is not None:
                pre_out["pred_masks"] = pred_segs
            out["pre_outputs"] = pre_out
            out["enc_meta"] = {"class_agnostic": False}

            if dn_meta is not None:
                dn_masks: list[Tensor | None] = (
                    list(dn_out_masks[:-1]) if dn_out_masks is not None else [None] * (self.num_layers - 1)
                )
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_masks,
                    dn_out_corners[-1] if dn_out_corners is not None else None,
                    dn_out_logits[-1] if dn_out_logits is not None else None,
                )
                dn_pre_out: dict[str, Any] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                }
                if dn_pre_segs is not None:
                    dn_pre_out["pred_masks"] = dn_pre_segs
                out["dn_pre_outputs"] = dn_pre_out
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: list[Tensor | None], outputs_coord: list[Tensor]) -> list[dict[str, Tensor]]:
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord) if a is not None]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class: Tensor | None,
        outputs_coord: Tensor | None,
        outputs_corners: Tensor | None,
        outputs_ref: Tensor | None,
        outputs_masks: list[Tensor | None] | None = None,
        teacher_corners: Tensor | None = None,
        teacher_logits: Tensor | None = None,
    ) -> list[dict[str, Any]]:
        if outputs_class is None or outputs_coord is None:
            return []
        results = []
        n = outputs_class.shape[0]
        for i in range(n):
            result: dict[str, Any] = {
                "pred_logits": outputs_class[i],
                "pred_boxes": outputs_coord[i],
                "pred_corners": outputs_corners[i] if outputs_corners is not None else None,
                "ref_points": outputs_ref[i] if outputs_ref is not None else None,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            if outputs_masks is not None and i < len(outputs_masks) and outputs_masks[i] is not None:
                result["pred_masks"] = outputs_masks[i]
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Postprocessor (used by ECDETRDetector.postprocess)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def postprocess(
        self,
        outputs: dict[str, Tensor],
        orig_target_sizes: Tensor,
        num_top_queries: int = 300,
    ) -> list[dict[str, Tensor]]:
        """Decode raw model outputs into boxes/labels/scores (and masks if present)."""
        import torchvision

        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        mask_pred = outputs.get("pred_masks")

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        # orig_target_sizes is (B, 2) with (H, W); xyxy needs (W, H, W, H) scale.
        bbox_pred = bbox_pred * orig_target_sizes.flip(-1).repeat(1, 2).unsqueeze(1)

        scores = F.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
        labels = index % self.num_classes
        index = index // self.num_classes
        boxes_out = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).expand(-1, -1, 4))

        masks_out: Tensor | None = None
        if mask_pred is not None:
            masks_out = mask_pred.gather(
                dim=1,
                index=index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, mask_pred.shape[-2], mask_pred.shape[-1]),
            )

        results = []
        for i, (s, lb, b) in enumerate(zip(scores, labels, boxes_out)):
            res: dict[str, Tensor] = {"scores": s, "labels": lb, "boxes": b}
            if masks_out is not None:
                h, w = orig_target_sizes[i].tolist()
                m = F.interpolate(
                    masks_out[i].unsqueeze(1), size=(int(h), int(w)), mode="bilinear", align_corners=False
                )
                res["masks"] = (m > 0.0).squeeze(1)
            results.append(res)
        return results
