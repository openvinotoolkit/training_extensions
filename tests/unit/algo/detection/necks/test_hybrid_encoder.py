# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of HybridEncoder."""

import torch
from otx.algo.detection.necks.hybrid_encoder import HybridEncoder


def test_hybrid_encoder_forward():
    hidden_dim = 256
    feat_strides = [8, 16, 32]
    in_channels = [128, 256, 512]
    encoder = HybridEncoder(in_channels=in_channels, hidden_dim=hidden_dim, feat_strides=feat_strides)

    # Create dummy input
    batch_size = 2
    input_sizes = [(128, 64, 64), (256, 32, 32), (512, 16, 16)]
    dummy_input = [
        torch.randn(batch_size, *input_sizes[0]),
        torch.randn(batch_size, *input_sizes[1]),
        torch.randn(batch_size, *input_sizes[2]),
    ]

    # Forward pass
    outputs = encoder(dummy_input)

    # Check output shapes
    assert len(outputs) == 3
    for i, output in enumerate(outputs):
        expected_shape = (batch_size, hidden_dim, input_sizes[i][1], input_sizes[i][2])
        assert output.shape == expected_shape
