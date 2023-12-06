# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of CustomAnchorGenerator."""

import pytest
import torch
from otx.algo.detection.heads.custom_anchor_generator import SSDAnchorGeneratorClustered


class TestSSDAnchorGeneratorClustered:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.anchor_generator = SSDAnchorGeneratorClustered(
            strides = (16, 32),
            widths = [
                [38.641007923271076, 92.49516032784699, 271.4234764938237, 141.53469410876247],
                [206.04136086566515, 386.6542727907841, 716.9892752215089, 453.75609561761405, 788.4629155558277],
            ],
            heights = [
                [48.9243877087132, 147.73088476194903, 158.23569788707474, 324.14510379107367],
                [587.6216059488938, 381.60024152086544, 323.5988913027747, 702.7486097568518, 741.4865860938451],
            ],
        )

    def test_gen_base_anchors(self) -> None:
        assert self.anchor_generator.base_anchors[0].shape == torch.Size([4, 4])
        assert self.anchor_generator.base_anchors[1].shape == torch.Size([5, 4])
