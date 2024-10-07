# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""MaskDINO transformer encoder module."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as f
from torch.nn.init import normal_

from otx.algo.detection.heads.rtdetr_decoder import MSDeformableAttention as MSDeformAttn
from otx.algo.instance_segmentation.heads.pixel_decoder.position_encoding import PositionEmbeddingSine
from otx.algo.instance_segmentation.layers.batch_norm import get_norm
from otx.algo.instance_segmentation.utils.utils import (
    Conv2d,
    ShapeSpec,
    c2_xavier_fill,
    get_clones,
)


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    """MSDeformAttnTransformerEncoderOnly is a transformer encoder with MSDeformable Attention.

    Args:
        d_model (int, optional): hidden dimension. Defaults to 256.
        nhead (int, optional): number of heads in the multi-head attention models. Defaults to 8.
        num_encoder_layers (int, optional): number of sub-encoder-layers in the encoder. Defaults to 6.
        dim_feedforward (int, optional): dimension of the feedforward network model. Defaults to 1024.
        dropout (float, optional): dropout value. Defaults to 0.1.
        num_feature_levels (int, optional): number of feature levels. Defaults to 4.
        enc_n_points (int, optional): number of points for MSDeformAttn. Defaults to 4.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_feature_levels: int = 4,
        enc_n_points: int = 4,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        # learnable position embedding for each feature level
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize the parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()  # noqa: SLF001
        normal_(self.level_embed)

    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        """Get the valid ratio of the mask."""
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_height / height
        valid_ratio_w = valid_width / width
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def _prepare_input(
        self,
        srcs: list[Tensor],
        pos_embeds: list[Tensor],
        masks: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prepare input for encoder."""
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds, strict=True)):
            _, _, h, w = src.shape
            spatial_shape = torch.tensor((h, w), device=src.device)
            spatial_shapes.append(spatial_shape)
            lvl_pos_embed = pos_embed.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
        return (
            torch.cat(src_flatten, 1),
            torch.cat(mask_flatten, 1),
            torch.cat(lvl_pos_embed_flatten, 1),
            torch.stack(spatial_shapes),
        )

    def forward(self, srcs: list[Tensor], pos_embeds: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the encoder."""
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]

        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = self._prepare_input(srcs, pos_embeds, masks)

        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]),
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    """MSDeformAttnTransformerEncoderLayer is a single layer of MSDeformable Attention Transformer.

    Args:
        d_model (int, optional): hidden dimension. Defaults to 256.
        d_ffn (int, optional): hidden dimension of feedforward network. Defaults to 1024.
        dropout (float, optional): dropout value. Defaults to 0.1.
        n_levels (int, optional): number of levels in MSDeformAttn. Defaults to 4.
        n_heads (int, optional): number of heads in MSDeformAttn. Defaults to 8.
        n_points (int, optional): number of points in MSDeformAttn. Defaults to 4.
    """

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

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        """Add position embedding to the tensor."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: Tensor) -> Tensor:
        """Forward pass of the feed forward network."""
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Forward pass of the encoder layer."""
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        return self.forward_ffn(src)


class MSDeformAttnTransformerEncoder(nn.Module):
    """MSDeformAttnTransformerEncoder is a stack of MSDeformAttnTransformerEncoderLayer.

    Args:
        encoder_layer (nn.ModuleList): encoder layer module list.
        num_layers (int): number of layers
    """

    def __init__(self, encoder_layer: nn.ModuleList, num_layers: int) -> None:
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes: Tensor, valid_ratios: Tensor, device: torch.device) -> Tensor:
        """Get reference points for the transformer encoder."""
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, device=device),
                torch.linspace(0.5, width - 0.5, width, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        pos: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Forward pass of the encoder."""
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, padding_mask)

        return output


class MaskDINOEncoder(nn.Module):
    """This is the multi-scale encoder in detection models, also named as pixel decoder in segmentation models.

    Args:
        input_shape (dict[str, ShapeSpec]): shapes (channels and stride) of the input features
        transformer_dropout (float): dropout probability in transformer
        transformer_nheads (int): number of heads in transformer
        transformer_dim_feedforward (int): dimension of feedforward network
        transformer_enc_layers (int): number of transformer encoder layers
        conv_dim (int): number of output channels for the intermediate conv layers.
        mask_dim (int): number of output channels for the final conv layer.
        norm (str): normalization for all conv layers
        num_feature_levels (int): feature scales used
        total_num_feature_levels (int): total feautre scales used (include the downsampled features)
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: str,
        transformer_in_features: list[str],
        common_stride: int,
        num_feature_levels: int,
        total_num_feature_levels: int,
    ):
        super().__init__()
        # this is the input shape of pixel decoder
        input_shape_list = sorted(input_shape.items(), key=lambda x: x[1].stride)  # type: ignore  # noqa: PGH003
        self.in_features = [k for k, v in input_shape_list]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape_list]
        self.feature_channels = [v.channels for k, v in input_shape_list]

        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        transformer_input_shape_list = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride, reverse=True)  # type: ignore  # noqa: PGH003
        self.transformer_in_features = [k for k, v in transformer_input_shape_list]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape_list]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape_list]

        self.maskdino_num_feature_levels = num_feature_levels  # always use 3 scales
        self.total_num_feature_levels = total_num_feature_levels
        self.common_stride = common_stride

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.low_resolution_index = transformer_in_channels.index(max(transformer_in_channels))
        self.high_resolution_index = 0
        if self.transformer_num_feature_levels > 1:
            input_proj_list = [
                nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )
                for in_channels in transformer_in_channels[::-1]
            ]

            # input projectino for downsample
            in_channels = max(transformer_in_channels)
            for _ in range(self.total_num_feature_levels - self.transformer_num_feature_levels):  # exclude the res2
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, conv_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, conv_dim),
                    ),
                )
                in_channels = conv_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    ),
                ],
            )

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.total_num_feature_levels,
        )
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        c2_xavier_fill(self.mask_features)
        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = max(int(np.log2(stride) - np.log2(self.common_stride)), 1)

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels,
                conv_dim,
                kernel_size=1,
                bias=use_bias,
                norm=lateral_norm,
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=f.relu,
            )
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            self.add_module(f"adapter_{idx + 1}", lateral_conv)
            self.add_module(f"layer_{idx + 1}", output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward_features(self, features: dict[str, Tensor]) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Forward pass of the encoder."""
        # backbone features
        srcs = []
        pos = []
        # additional downsampled features
        srcsl: list[Tensor] = []
        posl = []
        if self.total_num_feature_levels > self.transformer_num_feature_levels:
            smallest_feat = features[self.transformer_in_features[self.low_resolution_index]]
            _len_srcs = self.transformer_num_feature_levels
            for lvl in range(_len_srcs, self.total_num_feature_levels):
                src = self.input_proj[lvl](smallest_feat) if lvl == _len_srcs else self.input_proj[lvl](srcsl[-1])
                srcsl.append(src)
                posl.append(self.pe_layer(src))
        srcsl = srcsl[::-1]
        # Reverse feature maps
        for idx, feat in enumerate(self.transformer_in_features[::-1]):
            x = features[feat]
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        srcs.extend(srcsl)
        pos.extend(posl)
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.total_num_feature_levels
        for i in range(self.total_num_feature_levels):
            if i < self.total_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, feat in enumerate(self.in_features[: self.num_fpn_levels][::-1]):
            x = features[feat]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + f.interpolate(
                out[self.high_resolution_index],
                size=cur_fpn.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.total_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return self.mask_features(out[-1]), out[0], multi_scale_features
