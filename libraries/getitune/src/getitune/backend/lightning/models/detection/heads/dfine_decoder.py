# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""D-FINE Decoder. Modified from D-FINE (https://github.com/Peterande/D-FINE)."""

from __future__ import annotations

import copy
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, ClassVar

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
    get_contrastive_denoising_training_group,
)
from getitune.backend.lightning.models.common.utils.utils import inverse_sigmoid
from getitune.backend.lightning.models.detection.utils.utils import dfine_distance2bbox, dfine_weighting_function
from getitune.backend.lightning.models.utils.weight_init import bias_init_with_prob


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer with MSDeformableAttentionV2.

    Args:
        d_model (int): The number of expected features in the input. Defaults to 256.
        n_head (int): The number of heads in the multiheadattention models. Defaults to 8.
        dim_feedforward (int): The dimension of the feedforward network model. Defaults to 1024.
        dropout (float): The dropout value. Defaults to 0.0.
        activation (Callable[..., nn.Module] | None, optional): The activation function. Defaults to None.
        n_levels (int): The number of levels in MSDeformableAttention. Defaults to 4.
        num_points_list (list[int], optional): Number of distinct points for each layer. Defaults to [3, 6, 3].
    """

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
        n_levels: int = 4,
        num_points_list: list[int] = [3, 6, 3],  # noqa: B006
    ):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttentionV2(
            d_model,
            n_head,
            n_levels,
            num_points_list,
        )
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters of the model."""
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        """Add positional embedding to the input tensor."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: Tensor) -> Tensor:
        """Forward function of feed forward network."""
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        target: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: list[list[int]],
        attn_mask: Tensor | None = None,
        query_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """Forward function of the Transformer Decoder Layer.

        Args:
            target (Tensor): target feature tensor.
            reference_points (Tensor): reference points tensor.
            value (Tensor): value tensor.
            spatial_shapes (list[list[int]]): spatial shapes of the value tensor.
            attn_mask (Tensor | None, optional): attention mask. Defaults to None.
            query_pos_embed (Tensor | None, optional): query positional embedding. Defaults to None.

        Returns:
            Tensor: updated target tensor.
        """
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask, need_weights=False)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )

        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        return self.norm3(target.clamp(min=-65504, max=65504))


class TransformerDecoder(nn.Module):
    """Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.

    Args:
        hidden_dim (int): The number of expected features in the input.
        decoder_layer (nn.Module): The decoder layer module.
        decoder_layer_wide (nn.Module): The wide decoder layer module.
        num_layers (int): The number of layers.
        num_head (int): The number of heads in the multi-head attention models.
        reg_max (int): The number of discrete bins for bounding box regression.
        reg_scale (Tensor): The curvature of the Weighting Function.
        up (Tensor): The upper bound of the sequence.
        eval_idx (int, optional): evaluation index. Defaults to -1.
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: nn.Module,
        decoder_layer_wide: nn.Module,
        num_layers: int,
        num_head: int,
        reg_max: int,
        reg_scale: Tensor,
        up: Tensor,
        eval_idx: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)],
        )
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)])
        self.box_distance_weight = nn.Parameter(
            dfine_weighting_function(self.reg_max, self.up, self.reg_scale),
            requires_grad=False,
        )

    def value_op(
        self,
        memory: Tensor,
        memory_spatial_shapes: list[list[int]],
    ) -> tuple[Tensor, ...]:
        """Preprocess values for MSDeformableAttention."""
        memory = memory.reshape(memory.shape[0], memory.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return memory.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def forward(
        self,
        target: Tensor,
        ref_points_unact: Tensor,
        memory: Tensor,
        spatial_shapes: list[list[int]],
        bbox_head: nn.Module,
        score_head: nn.Module,
        query_pos_head: nn.Module,
        pre_bbox_head: nn.Module,
        integral: nn.Module,
        reg_scale: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward function of the Transformer Decoder.

        Args:
            target (Tensor): target feature tensor.
            ref_points_unact (Tensor): reference points tensor.
            memory (Tensor): memory tensor.
            spatial_shapes (list[list[int]]): spatial shapes of the memory tensor.
            bbox_head (nn.Module): bounding box head.
            score_head (nn.Module): label score head.
            query_pos_head (nn.Module): query position head.
            pre_bbox_head (nn.Module): pre-bounding box head.
            integral (nn.Module): integral module.
            reg_scale (Tensor): number of discrete bins for bounding box regression.
            attn_mask (Tensor | None, optional): attention mask tensor. Defaults to None.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                out_bboxes (Tensor): bounding box predictions from all layers
                out_logits (Tensor): label score predictions from all layers
                out_corners (Tensor): bounding box corner predictions from all layers
                out_refs (Tensor): reference points from all layers
                pre_bboxes (Tensor): initial bounding box predictions
                pre_scores (Tensor): initial label score predictions
        """
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, spatial_shapes)

        out_bboxes = []
        out_logits = []
        out_corners = []
        out_refs = []
        box_distance_weight = self.box_distance_weight

        ref_points_detach = f.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = f.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                initial_ref_boxes = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = dfine_distance2bbox(
                initial_ref_boxes,
                integral(pred_corners, box_distance_weight),
                reg_scale,
            )

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                out_logits.append(scores)
                out_bboxes.append(inter_ref_bbox)
                out_corners.append(pred_corners)
                out_refs.append(initial_ref_boxes)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(out_bboxes),  # out_bboxes
            torch.stack(out_logits),  # out_logits
            torch.stack(out_corners),  # out_corners
            torch.stack(out_refs),  # out_refs
            pre_bboxes,
            pre_scores,
        )


class DFINETransformerModule(nn.Module):
    """D-FINE Transformer Module.

    Args:
        num_classes (int, optional): num of classes. Defaults to 80.
        hidden_dim (int, optional): Hidden dimension size.. Defaults to 256.
        num_queries (int, optional): Number of queries. Defaults to 300.
        feat_channels (list[int], optional): List of feature channels. Defaults to [256, 256, 256].
        num_points_list (list[int], optional): Number of points for each level. Defaults to [3, 6, 3].
        num_decoder_layers (int, optional): Number of decoder layers. Defaults to 6.
        dim_feedforward (int, optional): Dimension of the feedforward network. Defaults to 1024.
        dropout (float, optional): dropout rate. Defaults to 0.0.
        activation (Callable[..., nn.Module], optional): activation layer. Defaults to nn.ReLU.
        num_denoising (int, optional): Number of denoising samples. Defaults to 100.
        label_noise_ratio (float, optional): Ratio of label noise. Defaults to 0.5.
        box_noise_scale (float, optional): Scale of box noise. Defaults to 1.0.
        eval_spatial_size (tuple[int, int], optional): Spatial size for evaluation. Defaults to (640, 640).
        eval_idx (int, optional): Evaluation index. Defaults to -1.
        reg_scale (float, optional): The weight curvature. Defaults to 4.0.
        reg_max (int, optional): The number of bins for box regression. Defaults to 32.
    """

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: list[int] = [256, 256, 256],  # noqa: B006
        feat_strides: list[int] = [8, 16, 32],  # noqa: B006
        num_levels: int = 3,
        num_points_list: list[int] = [3, 6, 3],  # noqa: B006
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        eval_spatial_size: tuple[int, int] = (640, 640),
        eval_idx: int = -1,
        reg_scale: float = 4.0,
        reg_max: int = 32,
    ):
        super().__init__()
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = 1e-2
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.reg_max = reg_max

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
            activation,
            num_levels,
            num_points_list,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points_list,
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_decoder_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
        )
        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.query_pos_head = MLP(
            input_dim=4,
            hidden_dim=2 * hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activation=partial(activation, inplace=True),
        )

        # encoder head
        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    ("norm", nn.LayerNorm(hidden_dim)),
                ],
            ),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=4,
            num_layers=3,
            activation=partial(activation, inplace=True),
        )

        # decoder head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_decoder_layers + eval_idx
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)])
        # distribution refinement over num of self.reg_max bins
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=4 * (self.reg_max + 1),
                    num_layers=3,
                    activation=partial(activation, inplace=True),
                )
                for _ in range(num_decoder_layers)
            ],
        )
        self.pre_bbox_head = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=4,
            num_layers=3,
            activation=partial(activation, inplace=True),
        )

        self.integral = Integral(self.reg_max)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        self._reset_parameters(feat_channels)

    def _reset_parameters(self, feat_channels: list[int]) -> None:
        """Reset parameters of the module."""
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

        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels: list[int]) -> None:
        """Build input projection layer."""
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
                            ],
                        ),
                    ),
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
                            ],
                        ),
                    ),
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: list[Tensor]) -> tuple[Tensor, list[list[int]]]:
        """Flatten feature maps and get spatial shapes for encoder input.

        Args:
            feats (list[Tensor]): List of feature maps.

        Returns:
            tuple[Tensor, list[list[int]]]:
                Tensor: Flattened feature maps.
                list[list[int]]: List of spatial shapes for each feature map.
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
        device: str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        tensor_anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((tensor_anchors > self.eps) * (tensor_anchors < 1 - self.eps)).all(-1, keepdim=True)
        tensor_anchors = torch.log(tensor_anchors / (1 - tensor_anchors))
        tensor_anchors = torch.where(valid_mask, tensor_anchors, torch.inf)

        return tensor_anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: Tensor,
        spatial_shapes: list[list[int]],
        denoising_logits: Tensor | None = None,
        denoising_bbox_unact: Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory,
            enc_outputs_logits,
            anchors,
            self.num_queries,
        )

        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = f.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

            content = enc_topk_memory.detach()
            content = enc_topk_memory.detach()

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
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select top-k memory, logits, and anchors.

        Args:
            memory (Tensor): memory tensor.
            outputs_logits (Tensor): logits tensor.
            outputs_anchors_unact (Tensor): unactivated anchors tensor.
            topk (int): number of top-k to select.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                Tensor: top-k memory tensor.
                Tensor: top-k logits tensor.
                Tensor: top-k anchors tensor.
        """
        _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]),
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
        feats: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        explain_mode: bool = False,
    ) -> dict[str, Tensor]:
        """Forward function of the D-FINE Decoder Transformer Module.

        Args:
            feats (Tensor): Feature maps.
            targets (list[dict[str, Tensor]] | None, optional): target annotations. Defaults to None.
            explain_mode (bool, optional): Whether to return raw logits for explanation. Defaults to False.

        Returns:
            dict[str, Tensor]: Output dictionary containing predicted logits, losses and boxes.
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

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            raw_logits,
        ) = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
        )

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            target=init_ref_contents,
            ref_points_unact=init_ref_points_unact,
            memory=memory,
            spatial_shapes=spatial_shapes,
            bbox_head=self.dec_bbox_head,
            score_head=self.dec_score_head,
            query_pos_head=self.query_pos_head,
            pre_bbox_head=self.pre_bbox_head,
            integral=self.integral,
            reg_scale=self.reg_scale,
            attn_mask=attn_mask,
        )

        out_bboxes = out_bboxes.clamp(min=1e-8)

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)

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
            out["aux_outputs"] = self._set_aux_loss2(
                outputs_class=out_logits[:-1],
                outputs_coord=out_bboxes[:-1],
                outputs_corners=out_corners[:-1],
                outputs_ref=out_refs[:-1],
                teacher_corners=out_corners[-1],
                teacher_logits=out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(
                enc_topk_logits_list,
                enc_topk_bboxes_list,
            )
            out["pre_outputs"] = {
                "pred_logits": pre_logits,
                "pred_boxes": pre_bboxes,
            }

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    outputs_class=dn_out_logits,
                    outputs_coord=dn_out_bboxes,
                    outputs_corners=dn_out_corners,
                    outputs_ref=dn_out_refs,
                    teacher_corners=dn_out_corners[-1],
                    teacher_logits=dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                }
                out["dn_meta"] = dn_meta
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
            }

        if explain_mode:
            out["raw_logits"] = raw_logits

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: Tensor, outputs_coord: Tensor) -> list[dict[str, Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class: Tensor,
        outputs_coord: Tensor,
        outputs_corners: Tensor,
        outputs_ref: Tensor,
        teacher_corners: Tensor,
        teacher_logits: Tensor,
    ) -> list[dict[str, Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
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


class DFINETransformer:
    """DFINETransformer factory for detection."""

    decoder_cfg: ClassVar[dict[str, Any]] = {
        "dfine_hgnetv2_n": {
            "feat_channels": [128, 128],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "dim_feedforward": 512,
            "num_levels": 2,
            "num_decoder_layers": 3,
            "eval_idx": -1,
            "num_points_list": [6, 6],
        },
        "dfine_hgnetv2_s": {
            "feat_channels": [256, 256, 256],
            "num_decoder_layers": 3,
            "eval_idx": -1,
            "num_points_list": [3, 6, 3],
        },
        "dfine_hgnetv2_m": {
            "num_decoder_layers": 4,
            "eval_idx": -1,
        },
        "dfine_hgnetv2_l": {},
        "dfine_hgnetv2_x": {
            "feat_channels": [384, 384, 384],
            "reg_scale": 8.0,
            "eval_idx": -1,
        },
        "deim_dfine_hgnetv2_n": {
            "feat_channels": [128, 128],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "dim_feedforward": 512,
            "num_levels": 2,
            "num_decoder_layers": 3,
            "eval_idx": -1,
            "num_points_list": [6, 6],
            "activation": nn.SiLU,
        },
        "deim_dfine_hgnetv2_s": {
            "feat_channels": [256, 256, 256],
            "num_decoder_layers": 3,
            "eval_idx": -1,
            "num_points_list": [3, 6, 3],
            "activation": nn.SiLU,
        },
        "deim_dfine_hgnetv2_m": {
            "num_decoder_layers": 4,
            "eval_idx": -1,
            "activation": nn.SiLU,
        },
        "deim_dfine_hgnetv2_l": {
            "activation": nn.SiLU,
        },
        "deim_dfine_hgnetv2_x": {
            "feat_channels": [384, 384, 384],
            "reg_scale": 8.0,
            "eval_idx": -1,
            "activation": nn.SiLU,
        },
    }

    def __new__(
        cls, model_name: str, num_classes: int, eval_spatial_size: tuple[int, int] = (640, 640)
    ) -> DFINETransformerModule:
        """Constructor for DFINETransformerModule."""
        cfg = cls.decoder_cfg[model_name]
        return DFINETransformerModule(num_classes=num_classes, eval_spatial_size=eval_spatial_size, **cfg)
