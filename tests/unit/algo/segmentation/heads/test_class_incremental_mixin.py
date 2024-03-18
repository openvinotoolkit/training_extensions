# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of class incremental mixin of OTX segmentation task."""

from typing import ClassVar

import torch
from otx.algo.segmentation.litehrnet import LiteHRNet


class MockGT:
    data = torch.randint(0, 2, (1, 512, 512))


class MockDataSample:
    metainfo: ClassVar = {
        "ignored_labels": [2],
        "img_shape": [512, 512],
        "scale_factor": (7.03125, 7.03125),
        "padding_size": (0, 0, 0, 0),
        "ori_shape": (128, 128),
        "pad_shape": (512, 512),
    }
    gt_sem_seg = MockGT()


class TestClassIncrementalMixin:
    def test_ignore_label(self) -> None:
        hrnet = LiteHRNet(3, "s")
        hrnet_head = hrnet.model.decode_head

        seg_logits = torch.randn(1, 3, 128, 128)
        batch_data_samples = [MockDataSample()]

        loss_with_ignored_labels = hrnet_head.loss_by_feat(seg_logits, batch_data_samples)["loss_ce_ignore"]

        batch_data_samples[0].metainfo["ignored_labels"] = None
        loss_without_ignored_labels = hrnet_head.loss_by_feat(seg_logits, batch_data_samples)["loss_ce_ignore"]

        assert loss_with_ignored_labels < loss_without_ignored_labels
