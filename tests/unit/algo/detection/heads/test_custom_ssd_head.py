# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of CustomSSDHead."""

from otx.algo.detection.heads.ssd_head import SSDHead
from otx.algo.detection.losses.cross_entropy_loss import CrossEntropyLoss


class TestSSDHead:
    def test_init(self, mocker) -> None:
        self.head = SSDHead(
            num_classes=80,
            in_channels=(96, 320),
            use_depthwise=True,
            anchor_generator={
                "strides": (16, 32),
                "widths": [[38, 92, 271, 141], [206, 386, 716, 453, 788]],
                "heights": [[48, 147, 158, 324], [587, 381, 323, 702, 741]],
            },
            init_cfg={"type": "Xavier", "layer": "Conv2d", "distribution": "uniform"},
            bbox_coder={
                "target_means": [0.0, 0.0, 0.0, 0.0],
                "target_stds": [0.1, 0.1, 0.1, 0.1],
            },
            train_cfg={
                "assigner": {
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

        assert isinstance(self.head.loss_cls, CrossEntropyLoss)
