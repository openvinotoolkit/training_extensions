# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of CustomSSDHead."""

from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from otx.algo.detection.heads.custom_ssd_head import SSDHead


class TestSSDHead:
    def test_init(self, mocker) -> None:
        self.head = SSDHead(
            num_classes=80,
            in_channels=(96, 320),
            use_depthwise=True,
            anchor_generator={
                "type": "SSDAnchorGeneratorClustered",
                "strides": (16, 32),
                "widths": [[38, 92, 271, 141], [206, 386, 716, 453, 788]],
                "heights": [[48, 147, 158, 324], [587, 381, 323, 702, 741]],
            },
            act_cfg={"type": "ReLU"},
            init_cfg={"type": "Xavier", "layer": "Conv2d", "distribution": "uniform"},
            bbox_coder={
                "type": "DeltaXYWHBBoxCoder",
                "target_means": [0.0, 0.0, 0.0, 0.0],
                "target_stds": [0.1, 0.1, 0.1, 0.1],
            },
        )

        assert isinstance(self.head.loss_cls, CrossEntropyLoss)
