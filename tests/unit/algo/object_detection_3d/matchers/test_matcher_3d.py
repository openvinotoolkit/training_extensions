# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test for HungarianMatcher3D module."""

import pytest
import torch
from otx.algo.object_detection_3d.matchers.matcher_3d import HungarianMatcher3D


class TestHungarianMatcher3D:
    @pytest.fixture()
    def matcher(self):
        return HungarianMatcher3D()

    def test_hungarian_matcher_3d(self, matcher):
        outputs = {
            "scores": torch.randn(1, 100, 10),
            "boxes_3d": torch.randn(1, 100, 6),
        }
        targets = [
            {
                "labels": torch.tensor([0, 0, 0, 0]),
                "boxes": torch.tensor(
                    [
                        [0.7697, 0.4923, 0.0398, 0.0663],
                        [0.7371, 0.4857, 0.0339, 0.0620],
                        [0.7126, 0.4850, 0.0246, 0.0501],
                        [0.5077, 0.5280, 0.0444, 0.1475],
                    ],
                ),
                "boxes_3d": torch.tensor(
                    [
                        [0.7689, 0.4918, 0.0191, 0.0208, 0.0327, 0.0336],
                        [0.7365, 0.4858, 0.0163, 0.0175, 0.0310, 0.0310],
                        [0.7122, 0.4848, 0.0118, 0.0127, 0.0248, 0.0252],
                        [0.5089, 0.5234, 0.0235, 0.0209, 0.0693, 0.0783],
                    ],
                ),
            },
        ]
        group_num = 11

        result = matcher(outputs, targets, group_num)

        assert len(result) == 1
        assert isinstance(result[0][0], torch.Tensor)
        assert isinstance(result[0][1], torch.Tensor)
        assert len(result[0][0].tolist()) == 44
        assert len(result[0][1].tolist()) == 44
        assert torch.max(torch.stack(result[0])) <= 100
