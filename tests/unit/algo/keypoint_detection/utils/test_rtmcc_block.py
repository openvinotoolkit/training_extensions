# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of RTMCCBlock."""

import torch
from otx.algo.keypoint_detection.utils.rtmcc_block import RTMCCBlock


class TestRTMCCBlock:
    def test_forward(self) -> None:
        """Test forward."""
        rtmcc_block = RTMCCBlock(
            num_token=10,
            in_token_dims=256,
            out_token_dims=256,
            s=128,
            expansion_factor=2,
            attn_type="self-attn",
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        )

        assert isinstance(rtmcc_block.act_fn, torch.nn.SiLU)
        assert rtmcc_block.uv.in_features == 256
        assert rtmcc_block.uv.out_features == 1152
        assert rtmcc_block.gamma.shape == (2, 128)
        assert rtmcc_block.beta.shape == (2, 128)

        inputs = torch.arange(256 * 256, dtype=torch.float32).view(1, 256, 256)
        result = rtmcc_block.forward(inputs)
        assert result.shape == inputs.shape
