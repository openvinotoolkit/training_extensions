# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.dense_heads.ssd_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/dense_heads/ssd_head.py
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from otx.algo.common.utils.coders import BaseBBoxCoder
from otx.algo.common.utils.prior_generators import BasePriorGenerator
from otx.algo.common.utils.samplers import PseudoSampler
from otx.algo.detection.heads.anchor_head import AnchorHead
from otx.core.data.entity.detection import DetBatchDataEntity


class SSDHeadModule(AnchorHead):
    """Implementation of `SSD head <https://arxiv.org/abs/1512.02325>`_.

    Args:
        anchor_generator (nn.Module): Config dict for anchor generator.
        bbox_coder (nn.Module): Config of bounding box coder.
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
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        test_cfg (dict, Optional): Testing config of anchor head.
        use_sigmoid_cls (bool): Whether to use a sigmoid activation function for
            classification prediction. Defaults to False.
    """

    def __init__(
        self,
        anchor_generator: nn.Module,
        bbox_coder: nn.Module,
        init_cfg: dict | list[dict],
        train_cfg: dict,
        num_classes: int = 80,
        in_channels: tuple[int, ...] | int = (512, 1024, 512, 256, 256, 256),
        stacked_convs: int = 0,
        feat_channels: int = 256,
        use_depthwise: bool = False,
        reg_decoded_bbox: bool = False,
        test_cfg: dict | None = None,
        use_sigmoid_cls: bool = False,
    ) -> None:
        super(AnchorHead, self).__init__(init_cfg=init_cfg, use_sigmoid_cls=use_sigmoid_cls)
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

        self.bbox_coder = bbox_coder
        self.reg_decoded_bbox = reg_decoded_bbox
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

    def prepare_loss_inputs(
        self,
        x: tuple[Tensor],
        entity: DetBatchDataEntity,
    ) -> dict | tuple:
        """Perform forward propagation of the detection head and prepare for loss calculation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            entity (DetBatchDataEntity): Entity from OTX dataset.

        Returns:
            dict: A dictionary of components for loss calculation.
        """
        (cls_scores, bbox_preds), batch_gt_instances, batch_img_metas = super().prepare_loss_inputs(x, entity)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
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
            "cls_score": all_cls_scores,
            "bbox_pred": all_bbox_preds,
            "anchor": all_anchors,
            "labels": all_labels,
            "label_weights": all_label_weights,
            "bbox_targets": all_bbox_targets,
            "bbox_weights": all_bbox_weights,
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


class SSDHead:
    """SSDHead factory for detection."""

    SSDHEAD_CFG: ClassVar[dict[str, Any]] = {
        "ssd_mobilenetv2": {
            "in_channels": (96, 320),
            "use_depthwise": True,
        },
    }

    def __new__(
        cls,
        model_name: str,
        num_classes: int,
        anchor_generator: BasePriorGenerator,
        bbox_coder: BaseBBoxCoder,
        init_cfg: dict,
        train_cfg: dict,
        test_cfg: dict | None = None,
    ) -> SSDHeadModule:
        """Constructor for SSDHead."""
        if model_name not in cls.SSDHEAD_CFG:
            msg = f"model type '{model_name}' is not supported"
            raise KeyError(msg)

        return SSDHeadModule(
            **cls.SSDHEAD_CFG[model_name],
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            init_cfg=init_cfg,  # TODO (sungchul, kirill): remove
            train_cfg=train_cfg,  # TODO (sungchul, kirill): remove
            test_cfg=test_cfg,  # TODO (sungchul, kirill): remove
            use_sigmoid_cls=False,  # use softmax cls
        )
