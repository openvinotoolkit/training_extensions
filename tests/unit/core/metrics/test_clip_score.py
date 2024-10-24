# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.core.metrics.clip_score import CLIPScore, ImageTextMeanAveragePrecision


@pytest.fixture()
def fxt_image_features() -> torch.Tensor:
    return torch.tensor(
        [
            [0.2673, 0.5345, 0.8018],
            [0.4558, 0.5698, 0.6838],
        ],
    )


@pytest.fixture()
def fxt_text_features() -> torch.Tensor:
    return torch.tensor(
        [
            [0.4558, 0.5698, 0.6838],
            [0.2673, 0.5345, 0.8018],
        ],
    )


class TestCLIPScore:
    def test_clip_score_update_and_compute_same_feature(self, fxt_image_features):
        metric = CLIPScore()

        # Update the metric with the dummy features (same features)
        metric.update(fxt_image_features, fxt_image_features)

        # Compute the metric
        result = metric.compute()

        # Check if the result is as expected
        expected_result = torch.tensor(100.0)
        assert torch.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    def test_clip_score_update_and_compute(self, fxt_image_features, fxt_text_features):
        metric = CLIPScore()

        # Update the metric with the dummy features
        metric.update(fxt_image_features, fxt_text_features)

        # Compute the metric
        result = metric.compute()

        # Check if the result is as expected
        expected_result = torch.tensor(97.4649)
        assert torch.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"


class TestImageTextMeanAveragePrecision:
    def test_image_text_map_update_and_compute(self, fxt_image_features, fxt_text_features):
        metric = ImageTextMeanAveragePrecision(k=2)

        # Update the metric with the dummy features
        metric.update(fxt_image_features, fxt_text_features)

        # Compute the metric
        result = metric.compute()

        # Check if the result is as expected
        expected_result = torch.tensor(0.5)
        assert torch.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    def test_image_text_map_update_and_compute_same_feature(self, fxt_image_features):
        metric = ImageTextMeanAveragePrecision(k=2)

        # Update the metric with the dummy features
        metric.update(fxt_image_features, fxt_image_features)

        # Compute the metric
        result = metric.compute()

        # Check if the result is as expected
        expected_result = torch.tensor(1.0)
        assert torch.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    def test_image_text_map_reset(self, fxt_image_features, fxt_text_features):
        metric = ImageTextMeanAveragePrecision(k=2)

        # Update the metric with the dummy features
        metric.update(fxt_image_features, fxt_text_features)

        result = metric.compute()
        assert result > 0, f"Expected > 0, but got {result}"

        # Reset the metric
        metric.reset()

        # Check if the metric is reset correctly
        assert metric.average_precision == torch.tensor(0.0), f"Expected 0.0, but got {metric.average_precision}"
        assert metric.total == torch.tensor(0), f"Expected 0, but got {metric.total}"
