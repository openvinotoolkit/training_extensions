# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of class incremental mixin of OTX segmentation task."""


import torch
from otx.algo.segmentation.litehrnet import LiteHRNet
from otx.core.data.entity.base import ImageInfo


class MockGT:
    data = torch.randint(0, 2, (1, 512, 512))


class TestClassIncrementalMixin:
    def test_ignore_label(self) -> None:
        hrnet = LiteHRNet(3, input_size=(128, 128), model_version="lite_hrnet_18")

        seg_logits = torch.randn(1, 3, 128, 128)
        # no annotations for class=3
        masks = torch.randint(0, 2, (1, 128, 128))
        batch_data_samples = [ImageInfo(0, (128, 128), (128, 128), ignored_labels=[2])]

        loss_with_ignored_labels = hrnet.model(seg_logits, batch_data_samples, masks=masks, mode="loss")[
            "loss_ce_ignore"
        ]

        batch_data_samples[0].ignored_labels = []
        loss_without_ignored_labels = hrnet.model(seg_logits, batch_data_samples, masks=masks, mode="loss")[
            "loss_ce_ignore"
        ]

        assert loss_with_ignored_labels < loss_without_ignored_labels
