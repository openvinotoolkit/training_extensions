"""Custom patch of multi_scale_deformable_attn_pytorch for openvino export."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from typing import Optional

import mmcv
import torch
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.ops import multi_scale_deform_attn
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from torch import nn


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Custom patch for multi_scale_deformable_attn_pytorch function.

    Original implementation in mmcv.ops use torch.nn.functional.grid_sample.
    It raises errors during inference with OpenVINO exported model.
    Therefore this function change grid_sample function to _custom_grid_sample function.
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = _custom_grid_sample(
            value_l_,
            sampling_grid_l_,
            # mode='bilinear',
            # padding_mode='zeros',
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def _custom_grid_sample(im: torch.Tensor, grid: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
    """Custom patch for mmcv.ops.point_sample.bilinear_grid_sample.

    This function is almost same with mmcv.ops.point_sample.bilinear_grid_sample.
    The only difference is this function use reshape instead of view.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    device = im.device
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.reshape(n, -1)
    y = y.reshape(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode="constant", value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0).to(device), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0).to(device), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0).to(device), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0).to(device), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def ms_deform_attn_pytorch_key_aware(
    query,
    value,
    key,
    input_padding_mask,
    value_spatial_shapes,
    sampling_locations,
    key_proj,
    value_proj,
    query_proj,
    attention_weights_linear,
    add,
):
    """Key aware multi scale deformable attention module.

    This codes are modified from https://github.com/IDEA-Research/Lite-DETR/blob/main/models/\
    dino/ops/functions/ms_deform_attn_func.py

    Cythonize using cuda implementation might be needed for speed optimization.
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    key_list = key.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    sampling_key_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        key_l_ = key_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = _custom_grid_sample(value_l_, sampling_grid_l_, align_corners=False)
        sampling_value_list.append(sampling_value_l_)

        sampling_key_l__ = _custom_grid_sample(key_l_, sampling_grid_l_, align_corners=False)
        sampling_key_list.append(sampling_key_l__)
    key = torch.stack(sampling_key_list, dim=-2).flatten(-2)
    value = torch.stack(sampling_value_list, dim=-2).flatten(-2)

    key = key.permute(0, 2, 3, 1).flatten(0, 1)
    N_, Lq, DD_ = query.shape
    query = query_proj(query)
    query = query.view(N_, Lq, M_, DD_ // M_)
    query = query.permute(0, 2, 1, 3).flatten(0, 2)
    query = query.unsqueeze(-2)
    dk = query.size()[-1]

    attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
    attention_weights = F.softmax(attention_weights, -1)

    value = value.permute(0, 2, 3, 1).flatten(0, 1)

    output = attention_weights.matmul(value)
    output = output.squeeze(-2).view(N_, M_, Lq_, D_).permute(0, 2, 1, 3)
    output = output.flatten(2)

    return output.contiguous()


@ATTENTION.register_module()
class KeyAwareMultiScaleDeformableAttention(MultiScaleDeformableAttention):
    """An attention module used in Lite-DETR.

    `Lite DETR : An Interleaved Multi-Scale Encoder for Efficient DETR.
    <https://arxiv.org/pdf/2303.07335.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
        norm_cfg: Optional[dict] = None,
        init_cfg: Optional[mmcv.ConfigDict] = None,
        same_loc: bool = False,
        proj_key: bool = True,
        add: bool = True,
    ):
        super().__init__(
            embed_dims, num_heads, num_levels, num_points, im2col_step, dropout, batch_first, norm_cfg, init_cfg
        )
        if same_loc:
            self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_points * 2)
        else:
            self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = None
        self.proj_key = proj_key
        if proj_key:
            self.key_proj = nn.Linear(embed_dims, embed_dims)
        else:
            self.key_proj = None
        self.query_proj = nn.Linear(embed_dims, embed_dims)
        self.add = add
        self.same_loc = same_loc
        self.init_additional_modules()

    def init_additional_modules(self) -> None:
        """Default initialization for Parameters of Module."""
        xavier_init(self.query_proj.weight.data)
        if self.proj_key:
            xavier_init(self.key_proj.weight.data)
        self._is_init = True

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward Function of Key aware MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            kwargs: Additional argumentations.

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if key is None:
            key = value

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if self.proj_key:
            key = self.key_proj(key)
        else:
            key = value
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
            key = key.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        key = key.view(bs, num_value, self.num_heads, -1)
        if not self.same_loc:
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
            )
        else:
            sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_points, 2)
            sampling_offsets = sampling_offsets[:, :, :, None].repeat(1, 1, 1, self.num_levels, 1, 1)

        attention_weights = None

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        output = ms_deform_attn_pytorch_key_aware(
            query,
            value,
            key,
            key_padding_mask,
            spatial_shapes,
            sampling_locations,
            self.key_proj,
            self.value_proj,
            self.query_proj,
            attention_weights,
            self.add,
        )

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


multi_scale_deform_attn.multi_scale_deformable_attn_pytorch = multi_scale_deformable_attn_pytorch
