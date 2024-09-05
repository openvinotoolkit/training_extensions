# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of common layers for detection task."""

import torch

from otx.algo.detection.layers.common_layer import AConv, ADown, Concat


class TestConcat:
    def test_forward(self):
        concat = Concat(dim=1)
        input_tensors = [torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)]
        output = concat(input_tensors)

        assert output.shape == (2, 6, 4, 4), "Output shape mismatch"
        assert torch.equal(output[:, :3, :, :], input_tensors[0]), "First part of concatenation mismatch"
        assert torch.equal(output[:, 3:, :, :], input_tensors[1]), "Second part of concatenation mismatch"


class TestAConv:
    def test_forward(self) -> None:
        aconv = AConv(in_channels=3, out_channels=6)
        input_tensor = torch.randn(2, 3, 4, 4)
        output = aconv(input_tensor)

        assert output.shape == (2, 6, 2, 2), "Output shape mismatch"


class TestADown:
    def test_forward(self):
        adown = ADown(in_channels=6, out_channels=12)
        input_tensor = torch.randn(2, 6, 4, 4)
        output = adown(input_tensor)

        assert output.shape == (2, 12, 2, 2), "Output shape mismatch"
