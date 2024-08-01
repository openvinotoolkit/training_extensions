# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of HungarianMatcher."""

import pytest
import torch
from otx.algo.detection.utils.matchers.hungarian_matcher import HungarianMatcher


class TestHungarianMatcher:
    @pytest.fixture()
    def targets(self):
        return [
            {
                "boxes": torch.tensor(
                    [
                        [0.6761, 0.8174, 0.7261, 0.8674],
                        [0.1652, 0.1109, 0.2152, 0.1609],
                        [0.2848, 0.9370, 0.3348, 0.9870],
                    ],
                ),
                "labels": torch.tensor([2, 2, 1]),
            },
        ]

    @pytest.fixture()
    def outputs(self):
        return {
            "pred_boxes": torch.tensor([[[0.17, 0.11, 0.21, 0.17], [0.5, 0.6, 0.7, 0.8], [0.3, 0.9, 0.3, 0.9]]]),
            "pred_logits": torch.tensor([[[0.9, 0.1, 0.3], [0.2, 0.8, 0.3], [0.3, 0.7, 0.1]]]),
        }

    def test_hungarian_matcher(self, targets, outputs):
        weight_dict = {
            "cost_class": 1.0,
            "cost_bbox": 1.0,
            "cost_giou": 1.0,
        }
        alpha = 0.25
        gamma = 2.0

        matcher = HungarianMatcher(weight_dict, alpha, gamma)

        # Perform the matching
        matches = matcher(outputs, targets)

        # Assert the output matches the expected shape
        assert len(matches) == 1
        assert all(len(match[0]) == len(match[1]) for match in matches)
        assert all(len(match[0]) == len(target["labels"]) for match, target in zip(matches, targets))
