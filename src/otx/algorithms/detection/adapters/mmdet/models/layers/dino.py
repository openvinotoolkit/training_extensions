"""Custom DINO transformer for OTX template."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Tuple, Union

import torch
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import DeformableDetrTransformer
from torch import Tensor, nn


@TRANSFORMER.register_module()
class CustomDINOTransformer(DeformableDetrTransformer):
    """Custom DINO transformer.

    Original implementation: mmdet.models.utils.transformer.DeformableDETR in mmdet2.x
    What's changed: The forward function is modified.
        Modified implementations come from mmdet.models.detectors.dino.DINO in mmdet3.x
    """

    def init_layers(self):
        """Initialize layers of the DINO.

        Unlike Deformable DETR, DINO does not need pos_trans, pos_trans_norm.
        """
        self.level_embeds = torch.nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

        self.enc_output = torch.nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = torch.nn.LayerNorm(self.embed_dims)

    def forward(
        self,
        batch_info: List[Dict[str, Union[Tuple, Tensor]]],
        mlvl_feats: List[Tensor],
        mlvl_masks: List[Tensor],
        query_embed: Tensor,
        mlvl_pos_embeds: List[Tensor],
        reg_branches: Optional[nn.ModuleList] = None,
        cls_branches: Optional[nn.ModuleList] = None,
        **kwargs
    ):
        """Forward function for `Transformer`.

        What's changed:
            In mmdet3.x forward of transformer is divided into
            pre_transformer() -> forward_encoder() -> pre_decoder() -> forward_decoder().
            In comparison, mmdet2.x forward function takes charge of all functions above.
            The differences in Deformable DETR and DINO are occured in pre_decoder(), forward_decoder().
            Therefore this function modified those parts. Modified implementations come from
            pre_decoder(), and forward_decoder() of mmdet.models.detectors.dino.DINO in mmdet3.x.


        Args:
            batch_info(list(dict(str, union(tuple, tensor)))):
                Information about batch such as image shape,
                gt information.
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.
            kwargs: Additional argument for forward_transformer function.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
                - dn_meta (Dict[str, int]): The dictionary saves information about
                    group collation, including 'num_denoising_queries' and
                    'num_denoising_groups'. It will be used for split outputs of
                    denoising and matching parts and loss calculation.
        """
        feat_flatten: Union[Tensor, List[Tensor]] = []
        mask_flatten: Union[Tensor, List[Tensor]] = []
        lvl_pos_embed_flatten: Union[Tensor, List[Tensor]] = []
        spatial_shapes: Union[Tensor, List[Tensor]] = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs
        )

        # pre_decoder part at mmdet 3.x version
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        cls_out_features = cls_branches[self.decoder.num_layers].out_features
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](output_memory) + output_proposals

        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=self.two_stage_num_proposals, dim=1)[1]
        topk_scores = torch.gather(enc_outputs_class, 1, topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = query_embed[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(batch_info)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        # forward_decoder part in mmdet 3.x
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=mask_flatten,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
        )

        if len(query) == self.two_stage_num_proposals:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        return inter_states, list(references), topk_scores, topk_coords, dn_meta
