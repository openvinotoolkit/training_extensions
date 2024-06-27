# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of CustomSSDHead."""

from omegaconf import DictConfig
from otx.algo.common.losses import CrossEntropyLoss
from otx.algo.common.utils.coders import DeltaXYWHBBoxCoder
from otx.algo.detection.heads import SSDHead
from otx.algo.detection.utils.prior_generators import SSDAnchorGeneratorClustered


class TestSSDHead:
    def test_init(self, mocker) -> None:
        train_cfg = DictConfig(
            {
                "assigner": {
                    "min_pos_iou": 0.0,
                    "ignore_iof_thr": -1,
                    "gt_max_assign_all": False,
                    "pos_iou_thr": 0.4,
                    "neg_iou_thr": 0.4,
                },
                "smoothl1_beta": 1.0,
                "allowed_border": -1,
                "pos_weight": -1,
                "neg_pos_ratio": 3,
                "debug": False,
                "use_giou": False,
                "use_focal": False,
            },
        )
        test_cfg = DictConfig(
            {
                "nms": {"type": "nms", "iou_threshold": 0.45},
                "min_bbox_size": 0,
                "score_thr": 0.02,
                "max_per_img": 200,
            },
        )
        self.head = SSDHead(
            anchor_generator=SSDAnchorGeneratorClustered(
                strides=[16, 32],
                widths=[
                    [38.641007923271076, 92.49516032784699, 271.4234764938237, 141.53469410876247],
                    [206.04136086566515, 386.6542727907841, 716.9892752215089, 453.75609561761405, 788.4629155558277],
                ],
                heights=[
                    [48.9243877087132, 147.73088476194903, 158.23569788707474, 324.14510379107367],
                    [587.6216059488938, 381.60024152086544, 323.5988913027747, 702.7486097568518, 741.4865860938451],
                ],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            num_classes=3,
            in_channels=(96, 320),
            use_depthwise=True,
            init_cfg={"type": "Xavier", "layer": "Conv2d", "distribution": "uniform"},
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        assert isinstance(self.head.loss_cls, CrossEntropyLoss)
