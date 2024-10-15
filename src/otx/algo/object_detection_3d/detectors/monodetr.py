# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MonoDetr core Pytorch detector."""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional

from otx.algo.common.layers.transformer_layers import MLP
from otx.algo.common.utils.utils import get_clones, inverse_sigmoid
from otx.algo.object_detection_3d.utils.utils import NestedTensor


# TODO (Kirill): make MonoDETR as a more general class
class MonoDETR(nn.Module):
    """This is the MonoDETR module that performs monocualr 3D object detection."""

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
        init_box: bool = False,
        group_num: int = 11,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ):
        """Initializes the model.

        Args:
            backbone (nn.Module): torch module of the backbone to be used. See backbone.py
            depthaware_transformer (nn.Module): depth-aware transformer architecture. See depth_aware_transformer.py
            depth_predictor (nn.Module): depth predictor module
            criterion (nn.Module): loss criterion module
            num_classes (int): number of object classes
            num_queries (int): number of object queries, ie detection slot. This is the maximal number of objects
                       DETR can detect in a single image. For KITTI, we recommend 50 queries.
            num_feature_levels (int): number of feature levels
            aux_loss (bool): True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine (bool): iterative bounding box refinement
            init_box (bool): True if the bounding box embedding layers should be initialized to zero
            group_num (int): number of groups for depth-aware bounding box embedding
            activation (Callable[..., nn.Module]): activation function to be applied to the output of the transformer
        """
        super().__init__()

        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.criterion = criterion
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3, activation=activation)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2, activation=activation)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2, activation=activation)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2, activation=activation)  # depth and deviation

        if init_box:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim * 2)

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
        self.num_classes = num_classes

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = depthaware_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = get_clones(self.class_embed, num_pred)
            self.bbox_embed = get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = get_clones(self.angle_embed, num_pred)
            self.depth_embed = get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

    def forward(
        self,
        images: Tensor,
        calibs: Tensor,
        img_sizes: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        mode: str = "predict",
    ) -> dict[str, Tensor]:
        """Forward method of the MonoDETR model.

        Args:
            images (list[Tensor]): images for each sample
            calibs (Tensor): camera matrices for each sample
            img_sizes (Tensor): image sizes for each sample
            targets (list[dict[Tensor]): ground truth boxes and labels for each
                sample
            mode (str): The mode of operation. Defaults to "predict".
        """
        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for i, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[i](src))
            masks.append(mask)

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for i in range(_len_srcs, self.num_feature_levels):
                src = self.input_proj[i](features[-1].tensors) if i == _len_srcs else self.input_proj[i](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight if self.training else self.query_embed.weight[: self.num_queries]

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
        )

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
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
            depth_map = functional.grid_sample(
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

        out = {"scores": outputs_class[-1], "boxes_3d": outputs_coord[-1]}
        out["size_3d"] = outputs_3d_dim[-1]
        out["depth"] = outputs_depth[-1]
        out["heading_angle"] = outputs_angle[-1]
        if mode == "export":
            out["scores"] = out["scores"].sigmoid()
            return out

        out["pred_depth_map_logits"] = pred_depth_map_logits

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class,
                outputs_coord,
                outputs_3d_dim,
                outputs_angle,
                outputs_depth,
            )

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
    ) -> list[dict[str, Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"scores": a, "boxes_3d": b, "size_3d": c, "heading_angle": d, "depth": e}
            for a, b, c, d, e in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_3d_dim[:-1],
                outputs_angle[:-1],
                outputs_depth[:-1],
            )
        ]
