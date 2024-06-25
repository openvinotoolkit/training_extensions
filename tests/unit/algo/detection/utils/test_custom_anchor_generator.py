# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of CustomAnchorGenerator."""

import pytest
import torch
from otx.algo.detection.utils.prior_generators import SSDAnchorGeneratorClustered


class TestSSDAnchorGeneratorClustered:
    @pytest.fixture()
    def anchor_generator(self) -> SSDAnchorGeneratorClustered:
        return SSDAnchorGeneratorClustered(
            strides=(16, 32),
            widths=[
                [38.641007923271076, 92.49516032784699, 271.4234764938237, 141.53469410876247],
                [206.04136086566515, 386.6542727907841, 716.9892752215089, 453.75609561761405, 788.4629155558277],
            ],
            heights=[
                [48.9243877087132, 147.73088476194903, 158.23569788707474, 324.14510379107367],
                [587.6216059488938, 381.60024152086544, 323.5988913027747, 702.7486097568518, 741.4865860938451],
            ],
        )

    def test_gen_base_anchors(self, anchor_generator) -> None:
        assert anchor_generator.base_anchors[0].shape == torch.Size([4, 4])
        assert anchor_generator.base_anchors[1].shape == torch.Size([5, 4])

    def test_sparse_priors(self, anchor_generator) -> None:
        assert anchor_generator.sparse_priors(torch.IntTensor([0]), [32, 32], 0, device="cpu").shape == torch.Size(
            [1, 4],
        )

    def test_grid_anchors(self, anchor_generator) -> None:
        out = anchor_generator.grid_anchors([(8, 8), (16, 16)], device="cpu")
        assert len(out) == 2
        assert out[0].shape == torch.Size([256, 4])
        assert out[1].shape == torch.Size([1280, 4])

    def test_repr(self, anchor_generator) -> None:
        assert "strides" in str(anchor_generator)
        assert "widths" in str(anchor_generator)
        assert "heights" in str(anchor_generator)
        assert "num_levels" in str(anchor_generator)
        assert "centers" in str(anchor_generator)
        assert "center_offset" in str(anchor_generator)
