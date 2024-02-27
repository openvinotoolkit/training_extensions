# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of Module for OTX custom metrices."""

import torch
from otx.core.metrics.fmeasure import FMeasure


class TestFMeasure:
    def test_fmeasure(self) -> None:
        """Check whether f1 score is same with OTX1.x version."""
        metric = FMeasure()
        metric.num_classes = 1
        preds = [
            {
                "boxes": torch.Tensor([[0.7, 0.6, 0.9, 0.6], [0.2, 0.5, 0.8, 0.6]]),
                "labels": torch.IntTensor([0, 0]),
                "scores": torch.Tensor([0.9, 0.8]),
            },
            {
                "boxes": torch.Tensor([[0.3, 0.4, 0.6, 0.6], [0.3, 0.3, 0.4, 0.5]]),
                "labels": torch.IntTensor([0, 0]),
                "scores": torch.Tensor([0.9, 0.8]),
            },
        ]
        targets = [
            {
                "boxes": torch.Tensor([[0.8, 0.6, 0.9, 0.7], [0.3, 0.5, 0.8, 0.7]]),
                "labels": torch.IntTensor([0, 0]),
            },
            {
                "boxes": torch.Tensor([[0.4, 0.4, 0.6, 0.6], [0.3, 0.3, 0.4, 0.4]]),
                "labels": torch.IntTensor([0, 0]),
            },
        ]

        metric.update(preds, targets)
        result = metric.compute()
        assert result["f1-score"] == 0.5
