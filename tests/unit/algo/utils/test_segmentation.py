# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import torch
from otx.algo.segmentation.modules import (
    IterativeAggregator,
    channel_shuffle,
    normalize,
)
from otx.algo.segmentation.modules.blocks import OnnxLpNormalization


def test_channel_shuffle():
    assert channel_shuffle(torch.randn([1, 24, 8, 8]), 4).shape == torch.Size([1, 24, 8, 8])


def test_onnx_lp_normalization():
    assert OnnxLpNormalization().forward(None, torch.randn([1, 24, 8, 8])).shape == torch.Size([1, 24, 8, 8])


def test_normalize(mocker):
    input_tensor = torch.randn([1, 24])
    assert normalize(input_tensor, dim=-1).shape == torch.Size([1, 24])

    mock_apply = mocker.patch.object(OnnxLpNormalization, "apply")
    mocker.patch("torch.onnx.is_in_onnx_export", return_value=True)
    normalize(input_tensor, dim=-1)
    mock_apply.assert_called_once_with(input_tensor, -1, 2, 1e-12)


def test_iterative_aggregator():
    input_tensor_list = [
        torch.randn(1, 2, 16, 16),
        torch.randn(1, 4, 8, 8),
    ]
    out = IterativeAggregator([2, 4]).forward(input_tensor_list)
    assert len(out) == 2
    assert out[0].shape == torch.Size([1, 2, 16, 16])
    assert out[1].shape == torch.Size([1, 2, 8, 8])
