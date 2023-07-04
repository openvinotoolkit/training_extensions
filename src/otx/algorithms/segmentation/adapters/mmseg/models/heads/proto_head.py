"""Prototype based head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat
from mmcv.runner import force_fp32
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from torch import nn

from otx.algorithms.segmentation.adapters.mmseg.models.losses import PixelPrototypeCELoss
from otx.algorithms.segmentation.adapters.mmseg.models.utils import (
    ProjectionHead,
    distributed_sinkhorn,
    momentum_update,
    trunc_normal_,
)


@HEADS.register_module()
class ProtoNet(BaseDecodeHead):
    """Prototype based head.

    This head introduce prototype view learning. Prediction is achieved
    by nonparametric nearest prototype retrieving. This allows our model
    to directly shape the pixel embedding space, by optimizing the arrangement
    between embedded pixels and anchored prototypes.
    This network was developed based on two articles: https://arxiv.org/abs/2203.15102
    and https://arxiv.org/abs/2210.04388

    Args:
            gamma (bool): parameter used for momentum update.
                Defines influence of past states during the prototypes update.
            num_prototype (int): number of prototypes per class.
            in_proto_channels (int): number of channels of the prototypes (dimension).
            num_classes (int): number of classes.
    """

    def __init__(self, gamma, num_prototype, in_proto_channels, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.gamma = gamma
        self.num_prototype = num_prototype
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.num_prototype, in_proto_channels), requires_grad=False
        )
        trunc_normal_(self.prototypes, std=0.02)
        self.avg_pool = nn.AdaptiveAvgPool2d(in_proto_channels)
        self.proj_head = ProjectionHead(in_proto_channels, in_proto_channels)
        self.feat_norm = nn.LayerNorm(in_proto_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        """Prototype learning algorithm."""
        pred_seg = torch.max(out_seg, 1)[1]
        mask = gt_seg == pred_seg.view(-1)

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()

        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)
            m_k = mask[gt_seg == k]
            c_k = _c[gt_seg == k, ...]
            m_k_tile = repeat(m_k, "n -> n tile", tile=self.num_prototype)
            m_q = q * m_k_tile  # n x self.num_prototype
            c_k_tile = repeat(m_k, "n -> n tile", tile=c_k.shape[-1])
            c_q = c_k * c_k_tile  # n x embedding_dim
            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim
            n = torch.sum(m_q, dim=0)
            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(
                    old_value=protos[k, n != 0, :], new_value=f[n != 0, :], momentum=self.gamma, debug=False
                )
                protos[k, n != 0, :] = new_value

        self.prototypes = nn.Parameter(F.normalize(protos, p=2, dim=-1), requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def forward(self, inputs, gt_semantic_seg):
        """Forward method."""
        c = self.proj_head(inputs)
        _c = rearrange(c, "b c h w -> (b h w) c")
        _c = self.feat_norm(_c)
        _c = F.normalize(_c, p=2, dim=-1)
        self.prototypes.data.copy_(F.normalize(self.prototypes, p=2, dim=-1))
        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum("nd,kmd->nmk", _c, self.prototypes)

        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=inputs.shape[0], h=inputs.shape[2])
        gt_seg = F.interpolate(gt_semantic_seg.float(), size=inputs.size()[2:], mode="nearest").view(-1)
        contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
        out_seg = F.interpolate(out_seg, size=gt_semantic_seg.shape[-2:], mode="bilinear")
        proto_out = {"out_seg": out_seg, "contrast_logits": contrast_logits, "contrast_target": contrast_target}

        return proto_out

    @force_fp32(
        apply_to=(
            "out_seg",
            "contrast_logits",
            "contrast_target",
        )
    )
    def losses(self, out_seg, contrast_logits, contrast_target, seg_label):
        """Computes loss function."""
        loss = dict()

        if not isinstance(self.loss_decode, PixelPrototypeCELoss):
            raise ValueError("decode loss should be PixelPrototypeCELoss")

        loss[self.loss_decode.loss_name] = self.loss_decode(out_seg, contrast_logits, contrast_target, seg_label)

        loss["acc_seg"] = accuracy(out_seg, seg_label.squeeze(1), ignore_index=self.ignore_index)

        return loss
