# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of ELAN related layers for detection task."""

import torch
from otx.algo.detection.layers.elan_layer import ELAN, SPPELAN, RepConv, RepNCSPBottleneck, RepNCSPELAN


class TestELAN:
    def test_forward(self):
        in_channels = 64
        out_channels = 128
        part_channels = 32
        process_channels = 16

        elan = ELAN(in_channels, out_channels, part_channels, process_channels=process_channels)

        # Create a random input tensor
        x = torch.randn(1, in_channels, 32, 32)

        # Forward pass
        output = elan(x)

        # Check output shape
        assert output.shape == (1, out_channels, 32, 32)


class TestRepConv:
    def test_forward(self):
        in_channels = 64
        out_channels = 128
        kernel_size = 3

        rep_conv = RepConv(in_channels, out_channels, kernel_size)

        # Create a random input tensor
        x = torch.randn(1, in_channels, 32, 32)

        # Forward pass
        output = rep_conv(x)

        # Check output shape
        assert output.shape == (1, out_channels, 32, 32)


class TestRepNCSPBottleneck:
    def test_forward(self):
        in_channels = 64
        out_channels = 128
        kernel_size = (3, 3)
        residual = True
        expand = 1.0

        bottleneck = RepNCSPBottleneck(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            residual=residual,
            expand=expand,
        )

        # Create a random input tensor
        x = torch.randn(1, in_channels, 32, 32)

        # Forward pass
        output = bottleneck(x)

        # Check output shape
        assert output.shape == (1, out_channels, 32, 32)


class TestRepNCSPELAN:
    def test_forward(self):
        in_channels = 64
        out_channels = 128
        part_channels = 32
        process_channels = 16

        rep_ncsp_elan = RepNCSPELAN(in_channels, out_channels, part_channels, process_channels=process_channels)

        # Create a random input tensor
        x = torch.randn(1, in_channels, 32, 32)

        # Forward pass
        output = rep_ncsp_elan(x)

        # Check output shape
        assert output.shape == (1, out_channels, 32, 32)


class TestSPPELAN:
    def test_forward(self):
        in_channels = 64
        out_channels = 128
        neck_channels = 32

        spp_elan = SPPELAN(in_channels, out_channels, neck_channels)

        # Create a random input tensor
        x = torch.randn(1, in_channels, 32, 32)

        # Forward pass
        output = spp_elan(x)

        # Check output shape
        assert output.shape == (1, out_channels, 32, 32)
