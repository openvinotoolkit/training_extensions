# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MonoDetr core Pytorch detector."""
from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from otx.algo.common.utils.utils import inverse_sigmoid
from otx.algo.detection.heads.rtdetr_decoder import MLP
from otx.algo.object_detection_3d.utils.utils import NestedTensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MonoDETR(nn.Module):
    """This is the MonoDETR module that performs monocualr 3D object detection"""

    def __init__(
        self,
        backbone: nn.Module,
        depthaware_transformer: nn.Module,
        depth_predictor: nn.Module,
        criterion: nn.Module,
        num_classes: int,
        num_queries: int,
        num_feature_levels: int,
        aux_loss: bool = True,
        with_box_refine: bool = False,
        two_stage: bool = False,
        init_box: bool = False,
        use_dab: bool = False,
        group_num: int = 11,
        two_stage_dino: bool = False,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            depth_predictor: depth predictor module
            criterion: loss criterion module
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            num_feature_levels: number of feature levels
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
            init_box: True if the bounding box embedding layers should be initialized to zero
            use_dab: True if depth-aware bounding box embedding is to be used
            group_num: number of groups for depth-aware bounding box embedding
            two_stage_dino: True if two-stage MonoDETR with DINO is to be used
        """
        super().__init__()

        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.two_stage_dino = two_stage_dino
        self.criterion = criterion
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3, activation=nn.ReLU)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2, activation=nn.ReLU)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2, activation=nn.ReLU)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2, activation=nn.ReLU)  # depth and deviation
        self.use_dab = use_dab

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            if two_stage_dino:
                self.query_embed = None
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim * 2)
            else:
                self.tgt_embed = nn.Embedding(num_queries * group_num, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries * group_num, 6)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ),
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ),
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ),
                ],
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.num_classes = num_classes

        if self.two_stage_dino:
            _class_embed = nn.Linear(hidden_dim, num_classes)
            _bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3, activation=nn.ReLU)
            # init the two embed layers
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
            self.depthaware_transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.depthaware_transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (depthaware_transformer.decoder.num_layers + 1) if two_stage else depthaware_transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(
        self,
        images: list[Tensor],
        calibs: list[Tensor] | None = None,
        targets: list[dict[str, Tensor]] | None = None,
        img_sizes: list[Tensor] | None = None,
        mode: str = "predict",
    ) -> dict[str, Tensor] | Any:
        """Forward method of the MonoDETR model.

        Args:
            images (list[Tensor]): images for each sample
            calibs (list[Tensor]): camera matrices for each sample
            targets (list[dict[Tensor]): ground truth boxes and labels for each
                sample
            img_sizes (list[Tensor]): image sizes for each sample
            mode (str): The mode of operation. Defaults to "predict".
        """
        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.training:
                tgt_all_embed = tgt_embed = self.tgt_embed.weight  # nq, 256
                refanchor = self.refpoint_embed.weight  # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)

            else:
                tgt_all_embed = tgt_embed = self.tgt_embed.weight[: self.num_queries]
                refanchor = self.refpoint_embed.weight[: self.num_queries]
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
        elif self.two_stage_dino:
            query_embeds = None
        else:
            if self.training:
                query_embeds = self.query_embed.weight
            else:
                # only use one group in inference
                query_embeds = self.query_embed.weight[: self.num_queries]

        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(
            srcs,
            masks[1],
            pos[1],
        )

        (
            hs,
            init_reference,
            inter_references,
            inter_references_dim,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.depthaware_transformer(
            srcs,
            masks,
            pos,
            query_embeds,
            depth_pos_embed,
            depth_pos_embed_ip,
        )  # , attn_mask)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1:2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            # depth_reg
            depth_reg = self.depth_embed[lvl](hs[lvl])

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)

            # depth average + sigma
            depth_ave = torch.cat(
                [
                    ((1.0 / (depth_reg[:, :, 0:1].sigmoid() + 1e-6) - 1.0) + depth_geo.unsqueeze(-1) + depth_map) / 3,
                    depth_reg[:, :, 1:2],
                ],
                -1,
            )
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        out["pred_3d_dim"] = outputs_3d_dim[-1]
        out["pred_depth"] = outputs_depth[-1]
        out["pred_angle"] = outputs_angle[-1]
        out["pred_depth_map_logits"] = pred_depth_map_logits

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class,
                outputs_coord,
                outputs_3d_dim,
                outputs_angle,
                outputs_depth,
            )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {"pred_logits": enc_outputs_class, "pred_boxes": enc_outputs_coord}

        if mode == "loss":
            return self.criterion(outputs=out, targets=targets)

        return out

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class: Tensor,
        outputs_coord: Tensor,
        outputs_3d_dim: Tensor,
        outputs_angle: Tensor,
        outputs_depth: Tensor,
    ) -> dict[str, Tensor]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_3d_dim": c, "pred_angle": d, "pred_depth": e}
            for a, b, c, d, e in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_3d_dim[:-1],
                outputs_angle[:-1],
                outputs_depth[:-1],
            )
        ]
