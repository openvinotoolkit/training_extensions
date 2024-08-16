# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.dense_heads.ssd_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/dense_heads/ssd_head.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from otx.algo.common.utils.samplers import PseudoSampler
from otx.algo.detection.heads.anchor_head import AnchorHead

if TYPE_CHECKING:
    from otx.algo.utils.mmengine_utils import InstanceData


class SSDHead(AnchorHead):
    """Implementation of `SSD head <https://arxiv.org/abs/1512.02325>`_.

    Args:
        anchor_generator (nn.Module): Config dict for anchor generator.
        init_cfg (dict, list[dict]): Initialization config dict.
        train_cfg (dict): Training config of anchor head.
        num_classes (int): Number of categories excluding the background category.
        in_channels (Sequence[int]): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Defaults to 0.
        feat_channels (int): Number of hidden channels when stacked_convs > 0.
            Defaults to 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Defaults to False.
        test_cfg (dict, Optional): Testing config of anchor head.
    """

    def __init__(
        self,
        anchor_generator: nn.Module,
        init_cfg: dict | list[dict],
        train_cfg: dict,
        num_classes: int = 80,
        in_channels: tuple[int, ...] | int = (512, 1024, 512, 256, 256, 256),
        stacked_convs: int = 0,
        feat_channels: int = 256,
        use_depthwise: bool = False,
        test_cfg: dict | None = None,
    ) -> None:
        super(AnchorHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_depthwise = use_depthwise

        self.cls_out_channels = num_classes + 1  # add background class
        self.prior_generator = anchor_generator

        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = self.prior_generator.num_base_priors

        self._init_layers()

        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = self.train_cfg["assigner"]
            self.sampler = PseudoSampler(context=self)  # type: ignore[no-untyped-call]

    def forward(self, x: tuple[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            tuple[list[Tensor], list[Tensor]]: A tuple of cls_scores list and
            bbox_preds list.

            - cls_scores (list[Tensor]): Classification scores for all scale \
            levels, each is a 4D-tensor, the channels number is \
            num_anchors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale \
            levels, each is a 4D-tensor, the channels number is \
            num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(x, self.reg_convs, self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_gt_instances: list[InstanceData],
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | None = None,
    ) -> dict[str, Tensor]:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies deltas for each scale level with shape (N, num_anchors * 4, H, W)
            batch_gt_instances (list[InstanceData]): Batch of gt_instance.
                It usually includes ``bboxes`` and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[InstanceData], Optional): Batch of gt_instances_ignore.
                It includes ``bboxes`` attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of raw outputs.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            unmap_outputs=True,
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, avg_factor) = cls_reg_targets

        num_images = len(batch_img_metas)
        all_cls_scores = torch.cat(
            [s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in cls_scores],
            1,
        )
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        all_bbox_preds = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in bbox_preds], -2)
        all_bbox_targets = torch.cat(bbox_targets_list, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = [torch.cat(anchor) for anchor in anchor_list]

        return {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
            "all_anchors": all_anchors,
            "all_labels": all_labels,
            "all_label_weights": all_label_weights,
            "all_bbox_targets": all_bbox_targets,
            "all_bbox_weights": all_bbox_weights,
            "avg_factor": avg_factor,
        }

    def _init_layers(self) -> None:
        """Initialize layers of the head.

        This modificaiton is needed for smart weight loading
        """
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        if isinstance(self.in_channels, int):
            self.in_channels = (self.in_channels,)
        if isinstance(self.num_base_priors, int):
            self.num_base_priors = [self.num_base_priors]

        for in_channel, num_base_priors in zip(self.in_channels, self.num_base_priors):
            if self.use_depthwise:
                activation_layer = nn.ReLU(inplace=True)

                self.reg_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                        nn.BatchNorm2d(in_channel),
                        activation_layer,
                        nn.Conv2d(in_channel, num_base_priors * 4, kernel_size=1, padding=0),
                    ),
                )
                self.cls_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                        nn.BatchNorm2d(in_channel),
                        activation_layer,
                        nn.Conv2d(in_channel, num_base_priors * self.cls_out_channels, kernel_size=1, padding=0),
                    ),
                )
            else:
                self.reg_convs.append(nn.Conv2d(in_channel, num_base_priors * 4, kernel_size=3, padding=1))
                self.cls_convs.append(
                    nn.Conv2d(in_channel, num_base_priors * self.cls_out_channels, kernel_size=3, padding=1),
                )
