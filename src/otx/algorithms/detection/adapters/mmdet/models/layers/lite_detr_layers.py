"""Layers for Lite-DETR."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings

import torch
from mmcv.cnn import Linear, build_norm_layer
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import FFN, BaseTransformerLayer, build_transformer_layer
from mmcv.runner.base_module import BaseModule, Sequential
from torch import nn


@FEEDFORWARD_NETWORK.register_module()
class SmallExpandFFN(FFN):
    """Implements feed-forward networks (FFNs) with small expand.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(
            embed_dims,
            feedforward_channels,
            num_fcs,
            act_cfg,
            ffn_drop,
            dropout_layer,
            add_identity,
            init_cfg,
            **kwargs,
        )

        layers = []
        for _ in range(num_fcs - 1):
            layers.append(Sequential(Linear(embed_dims, embed_dims), self.activate, nn.Dropout(ffn_drop)))
        layers.append(Linear(embed_dims, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.small_expand_layers = Sequential(*layers)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x, level_start_index, enc_scale, identity=None):
        """Forward function for FFN."""
        x_3s = x[level_start_index[4 - enc_scale] :]
        x_4s = x[: level_start_index[4 - enc_scale]]
        x_4s = self.forward_ffn(self.small_expand_layers, self.norm2, x_4s, identity)
        x_3s = self.forward_ffn(self.layers, self.norm1, x_3s, identity)
        x = torch.cat([x_4s, x_3s], 0)

        return x

    def forward_ffn(self, layers, norm, x, identity=None):
        """Forward Feed Forward Network given layers."""
        out = layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return norm(identity + self.dropout_layer(out))


@TRANSFORMER_LAYER.register_module()
class EfficientTransformerLayer(BaseTransformerLayer):
    """Efficient TransformerLayer for Lite-DETR.

    It is base transformer encoder layer for Lite-DETR <https://arxiv.org/pdf/2303.07335.pdf>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        enc_scale (int): Scale of high level features. Default is 3.
    """

    def __init__(
        self,
        small_expand=False,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        enc_scale=3,
        **kwargs,
    ):

        super().__init__(attn_cfgs, ffn_cfgs, operation_order, norm_cfg, init_cfg, batch_first, **kwargs)
        self.enc_scale = enc_scale
        self.small_expand = small_expand

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
            level_start_index (Tensor): Start index for each level.
            kwargs: Additional arguments.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                if self.small_expand:
                    query = self.ffns[ffn_index](
                        query, level_start_index, self.enc_scale, identity if self.pre_norm else None
                    )
                else:
                    query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class EfficientTransformerEncoder(BaseModule):
    """TransformerEncoder of Lite-DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(
        self,
        transformerlayers=None,
        num_layers=None,
        init_cfg=None,
        post_norm_cfg=dict(type="LN"),
        enc_scale=3,
        num_expansion=3,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if len(transformerlayers) == 2 and num_layers != 2:
            if num_expansion == 1:
                _transformerlayers = [copy.deepcopy(transformerlayers[0]) for _ in range(num_layers - 1)] + [
                    transformerlayers[1]
                ]
            else:
                _transformerlayers = []
                for i in range(num_expansion):
                    for j in range(int(num_layers / num_expansion) - 1):
                        _transformerlayers.append(copy.deepcopy(transformerlayers[0]))
                    _transformerlayers.append(copy.deepcopy(transformerlayers[1]))
        else:
            assert isinstance(transformerlayers, list) and len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for layer in _transformerlayers:
            layer = build_transformer_layer(layer)
            assert layer.enc_scale == enc_scale
            self.layers.append(layer)
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        self.num_expansion = num_expansion
        self.enc_scale = enc_scale

        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f"Use prenorm in " f"{self.__class__.__name__}," f"Please specify post_norm_cfg"
            self.post_norm = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        level_start_index=None,
        reference_points=None,
        **kwargs,
    ):
        """Forward function for `TransformerCoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_queries, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_keys, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_keys, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in self-attention
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
            level_start_index (Tensor): Start index for each level.
            reference_points (Tensor): BBox predictions' reference.
            kwargs: Additional arguments.

        Returns:
            Tensor:  results with shape [num_queries, bs, embed_dims].
        """
        value = query
        value_tgt = value[level_start_index[4 - self.enc_scale] :]
        query = value_tgt
        reference_points_tgt = reference_points[:, level_start_index[4 - self.enc_scale] :]
        query_pos_tgt = query_pos[level_start_index[4 - self.enc_scale] :]
        for layer_id, layer in enumerate(self.layers):
            if (layer_id + 1) % (self.num_layers / self.num_expansion) == 0:
                query = value
                output = layer(
                    query,
                    key,
                    value,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    level_start_index=level_start_index,
                    key_pos=key_pos,
                    attn_masks=attn_masks,
                    query_key_padding_mask=query_key_padding_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                query = output[level_start_index[4 - self.enc_scale] :]
                value = output
            else:
                output = layer(
                    query,
                    key,
                    value,
                    query_pos=query_pos_tgt,
                    reference_points=reference_points_tgt,
                    level_start_index=level_start_index,
                    key_pos=key_pos,
                    attn_masks=attn_masks,
                    query_key_padding_mask=query_key_padding_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                query = output
                value = torch.cat([value[: level_start_index[4 - self.enc_scale]], query], 0)
        if self.post_norm is not None:
            output = self.post_norm(output)
        return value
