# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for PCK metric."""

from __future__ import annotations

import pytest
import torch
from otx.core.metrics.pck import PCKMeasure
from otx.core.types.label import LabelInfo


class TestPCK:
    @pytest.fixture()
    def fxt_preds(self) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "keypoints": torch.Tensor([[0.7, 0.6], [0.9, 0.6]]),
                "scores": torch.Tensor([0.9, 0.8]),
            },
            {
                "keypoints": torch.Tensor([[0.3, 0.4], [0.6, 0.6]]),
                "scores": torch.Tensor([0.9, 0.8]),
            },
        ]

    @pytest.fixture()
    def fxt_targets(self) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "keypoints": torch.Tensor([[0.3, 0.4], [0.6, 0.6]]),
                "keypoints_visible": torch.Tensor([0.9, 0.8]),
            },
            {
                "keypoints": torch.Tensor([[0.7, 0.6], [0.9, 0.6]]),
                "keypoints_visible": torch.Tensor([0.9, 0.8]),
            },
        ]

    def test_pck(self, fxt_preds, fxt_targets) -> None:
        metric = PCKMeasure(label_info=LabelInfo.from_num_classes(1))
        metric.input_size = (1, 1)
        metric.update(fxt_preds, fxt_targets)
        result = metric.compute()
        assert result["PCK"] == 0

        metric.reset()
        assert metric.preds == []
        assert metric.targets == []
