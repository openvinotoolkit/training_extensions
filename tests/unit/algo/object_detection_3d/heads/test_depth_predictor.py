# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test DepthPredictor."""

import pytest
import torch
from otx.algo.object_detection_3d.heads.depth_predictor import DepthPredictor


class TestDepthPredictor:
    @pytest.fixture()
    def depth_predictor(self):
        return DepthPredictor(depth_num_bins=10, depth_min=0.0, depth_max=1.0, hidden_dim=256)

    def test_depth_predictor_forward(self, depth_predictor):
        feature = [
            torch.randn(1, 256, 48, 160),
            torch.randn(1, 256, 24, 80),
            torch.randn(1, 256, 12, 40),
            torch.randn(1, 256, 6, 20),
        ]
        mask = torch.randn(1, 24, 80)
        pos = torch.randn(1, 256, 24, 80)

        depth_logits, depth_embed, weighted_depth, depth_pos_embed_ip = depth_predictor(feature, mask, pos)

        assert depth_logits.shape == (1, 11, 24, 80)
        assert depth_embed.shape == (1, 256, 24, 80)
        assert weighted_depth.shape == (1, 24, 80)
        assert depth_pos_embed_ip.shape == (1, 256, 24, 80)

    def test_depth_predictor_interpolate_depth_embed(self, depth_predictor):
        depth = torch.randn(1, 8, 8)
        interpolated_depth_embed = depth_predictor.interpolate_depth_embed(depth)

        assert interpolated_depth_embed.shape == (1, 256, 8, 8)

    def test_depth_predictor_interpolate_1d(self, depth_predictor):
        coord = torch.randn(1, 8, 8).clamp(min=0, max=1)
        interpolated_embeddings = depth_predictor.interpolate_1d(coord, depth_predictor.depth_pos_embed)

        assert interpolated_embeddings.shape == (1, 8, 8, 256)
