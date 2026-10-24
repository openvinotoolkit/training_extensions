# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for multilabel mAP metric."""

from __future__ import annotations

import pytest
import torch

from getitune.metrics.mlc_map import MultilabelmAP
from getitune.types.label import LabelInfo


class TestMAP:
    @pytest.fixture
    def fxt_preds(self) -> list[torch.Tensor]:
        return [torch.Tensor([0.7, 0.6, 0.1]), torch.Tensor([0.1, 0.6, 0.1])]

    @pytest.fixture
    def fxt_targets(self) -> list[torch.Tensor]:
        return [torch.Tensor([0, 0, 1]), torch.Tensor([0, 1, 0])]

    def test_mlc_map(self, fxt_preds, fxt_targets) -> None:
        metric = MultilabelmAP(label_info=LabelInfo.from_num_classes(3))
        metric.update(fxt_preds, fxt_targets)
        result = metric.compute()
        assert result["map"] > 0

        metric.reset()
        assert metric.preds == []
        assert metric.targets == []
