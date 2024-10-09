# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""test depth aware transformer head for 3d object detection."""

import pytest
import torch
from otx.algo.object_detection_3d.heads.depthaware_transformer import (
    DepthAwareTransformerBuilder,
    gen_sineembed_for_position,
)


class TestDepthAwareTransformer:
    @pytest.fixture()
    def depth_aware_transformer(self):
        return DepthAwareTransformerBuilder("monodetr_50")

    def test_depth_aware_transformer_forward(self, depth_aware_transformer):
        # Create dummy input tensors
        srcs = [
            torch.randn(1, 256, 48, 160),
            torch.randn(1, 256, 24, 80),
            torch.randn(1, 256, 12, 40),
            torch.randn(1, 256, 6, 20),
        ]
        masks = [
            torch.randn(1, 48, 160) < 0,
            torch.randn(1, 24, 80) < 0,
            torch.randn(1, 12, 40) < 0,
            torch.randn(1, 6, 20) < 0,
        ]
        pos_embeds = [
            torch.randn(1, 256, 48, 160),
            torch.randn(1, 256, 24, 80),
            torch.randn(1, 256, 12, 40),
            torch.randn(1, 256, 6, 20),
        ]
        query_embed = torch.randn(550, 512)
        depth_pos_embed = torch.randn(1, 256, 24, 80)
        depth_pos_embed_ip = torch.randn(1, 256, 24, 80)
        attn_mask = None
        depth_aware_transformer.decoder.return_intermediate = False
        output = depth_aware_transformer.forward(
            srcs,
            masks,
            pos_embeds,
            query_embed,
            depth_pos_embed,
            depth_pos_embed_ip,
            attn_mask,
        )

        # Check output shape
        assert len(output) == 6
        assert output[0].shape == (1, 550, 256)
        assert output[2].shape == (1, 550, 2)
        assert output[4] is None

    def test_depth_aware_transformer_get_proposal_pos_embed(self, depth_aware_transformer):
        # Create dummy input tensor
        proposals = torch.randn(2, 10, 6)

        # Get proposal position embeddings
        pos_embed = depth_aware_transformer.get_proposal_pos_embed(proposals)

        # Check output shape
        assert pos_embed.shape == (2, 10, 768)

    def test_depth_aware_transformer_get_valid_ratio(self, depth_aware_transformer):
        # Create dummy input tensor
        mask = torch.randn(2, 32, 32) > 0

        # Get valid ratio
        valid_ratio = depth_aware_transformer.get_valid_ratio(mask)

        # Check output shape
        assert valid_ratio.shape == (2, 2)

    def test_gen_sineembed_for_position(self):
        # Create dummy input tensor
        pos_tensor = torch.randn(2, 4, 6)

        # Generate sine embeddings for position tensor
        pos_embed = gen_sineembed_for_position(pos_tensor)

        # Check output shape
        assert pos_embed.shape == (2, 4, 768)
