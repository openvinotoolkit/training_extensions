# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Copy from https://github.com/open-mmlab/mmpretrain/blob/main/tests/test_models/test_utils/test_swiglu_ffn.py."""

import torch
from otx.algo.classification.utils.swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from torch import nn


class TestSwiGLUFFN:
    def test_init(self):
        swiglu = SwiGLUFFN(embed_dims=4)
        assert swiglu.w12.weight.shape == torch.ones((8, 4)).shape
        assert swiglu.w3.weight.shape == torch.ones((4, 4)).shape
        assert isinstance(swiglu.gamma2, nn.Identity)

    def test_forward(self):
        swiglu = SwiGLUFFN(embed_dims=4)
        x = torch.randn((1, 8, 4))
        out = swiglu(x)
        assert out.size() == x.size()

        swiglu = SwiGLUFFN(embed_dims=4, out_dims=12)
        x = torch.randn((1, 8, 4))
        out = swiglu(x)
        assert tuple(out.size()) == (1, 8, 12)


class TestSwiGLUFFNFused:
    def test_init(self):
        swiglu = SwiGLUFFNFused(embed_dims=4)
        assert swiglu.w12.weight.shape == torch.ones((16, 4)).shape
        assert swiglu.w3.weight.shape == torch.ones((4, 8)).shape
        assert isinstance(swiglu.gamma2, nn.Identity)

    def test_forward(self):
        swiglu = SwiGLUFFNFused(embed_dims=4)
        x = torch.randn((1, 8, 4))
        out = swiglu(x)
        assert out.size() == x.size()

        swiglu = SwiGLUFFNFused(embed_dims=4, out_dims=12)
        x = torch.randn((1, 8, 4))
        out = swiglu(x)
        assert tuple(out.size()) == (1, 8, 12)
