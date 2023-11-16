"""Unit tests of CustomFCNMaskHead for OTX template."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from mmengine.config import Config

from otx.v2.adapters.torch.mmengine.mmdet.modules.models.heads import (
    CustomFCNMaskHead,
)


class TestCustomFCNMaskHead:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.head = CustomFCNMaskHead(
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_mask=True,
                loss_weight=1.0
            ),
        )

    def test_predict_by_feat_single(self):
        """Test _predict_by_feat_single function."""
        mask_preds = torch.randn(100, 5, 28, 28)
        bboxes = torch.randn(100, 4)
        labels = torch.zeros(100, dtype=torch.int64)
        img_meta = {
            "img_shape": (1024, 1024),
            "batch_input_shape": (1024, 1024),
            "ori_shape": (1365, 2048, 3),
            "scale_factor": (0.5, 0.7501831501831502),
            "pad_shape": (32, 1024),
        }
        rcnn_test_cfg = Config(
            {
                "score_thr": 0.05,
                "nms": {"type": "nms", "iou_threshold": 0.5, "max_num": 100},
                "max_per_img": 100,
                "mask_thr_binary": 0.5,
            }
        )
        rescale = True
        activate_map = False
        out = self.head._predict_by_feat_single(
            mask_preds, bboxes, labels, img_meta, rcnn_test_cfg, rescale, activate_map
        )
        assert len(out) == 100
        assert out[0].shape == (1, 28, 28)
