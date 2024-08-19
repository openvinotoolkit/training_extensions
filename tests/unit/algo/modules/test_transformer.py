# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmcv/blob/main/tests/test_cnn/test_transformer.py

import pytest
import torch
from otx.algo.modules.drop import DropPath
from otx.algo.modules.transformer import FFN, AdaptivePadding, PatchEmbed
from torch import nn


def test_adaptive_padding():
    for padding in ("same", "corner"):
        kernel_size = 16
        stride = 16
        dilation = 1
        inputs = torch.rand(1, 1, 15, 17)
        adap_pad = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        out = adap_pad(inputs)
        # padding to divisible by 16
        assert (out.shape[2], out.shape[3]) == (16, 32)
        inputs = torch.rand(1, 1, 16, 17)
        out = adap_pad(inputs)
        # padding to divisible by 16
        assert (out.shape[2], out.shape[3]) == (16, 32)

        kernel_size = (2, 2)
        stride = (2, 2)
        dilation = (1, 1)

        adap_pad = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        inputs = torch.rand(1, 1, 11, 13)
        out = adap_pad(inputs)
        # padding to divisible by 2
        assert (out.shape[2], out.shape[3]) == (12, 14)

        kernel_size = (2, 2)
        stride = (10, 10)
        dilation = (1, 1)

        adap_pad = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        inputs = torch.rand(1, 1, 10, 13)
        out = adap_pad(inputs)
        #  no padding
        assert (out.shape[2], out.shape[3]) == (10, 13)

        kernel_size = (11, 11)
        adap_pad = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        inputs = torch.rand(1, 1, 11, 13)
        out = adap_pad(inputs)
        #  all padding
        assert (out.shape[2], out.shape[3]) == (21, 21)

        # test padding as kernel is (7,9)
        inputs = torch.rand(1, 1, 11, 13)
        stride = (3, 4)
        kernel_size = (4, 5)
        dilation = (2, 2)
        adap_pad = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        dilation_out = adap_pad(inputs)
        assert (dilation_out.shape[2], dilation_out.shape[3]) == (16, 21)
        kernel_size = (7, 9)
        dilation = (1, 1)
        adap_pad = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        kernel79_out = adap_pad(inputs)
        assert (kernel79_out.shape[2], kernel79_out.shape[3]) == (16, 21)
        assert kernel79_out.shape == dilation_out.shape

    # assert only support "same" "corner"
    with pytest.raises(AssertionError):
        AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=1)


def test_patch_embed():
    _b = 2
    _h = 3
    _w = 4
    _c = 3
    embed_dims = 10
    kernel_size = 3
    stride = 1
    dummy_input = torch.rand(_b, _c, _h, _w)
    patch_merge_1 = PatchEmbed(
        in_channels=_c,
        embed_dims=embed_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=1,
        normalization=None,
    )

    x1, shape = patch_merge_1(dummy_input)
    # test out shape
    assert x1.shape == (2, 2, 10)
    # test outsize is correct
    assert shape == (1, 2)
    # test L = out_h * out_w
    assert shape[0] * shape[1] == x1.shape[1]

    _b = 2
    _h = 10
    _w = 10
    _c = 3
    embed_dims = 10
    kernel_size = 5
    stride = 2
    dummy_input = torch.rand(_b, _c, _h, _w)
    # test dilation
    patch_merge_2 = PatchEmbed(
        in_channels=_c,
        embed_dims=embed_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        normalization=None,
    )

    x2, shape = patch_merge_2(dummy_input)
    # test out shape
    assert x2.shape == (2, 1, 10)
    # test outsize is correct
    assert shape == (1, 1)
    # test L = out_h * out_w
    assert shape[0] * shape[1] == x2.shape[1]

    stride = 2
    input_size = (10, 10)

    dummy_input = torch.rand(_b, _c, _h, _w)
    # test stride and norm
    patch_merge_3 = PatchEmbed(
        in_channels=_c,
        embed_dims=embed_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        normalization=nn.LayerNorm,
        input_size=input_size,
    )

    x3, shape = patch_merge_3(dummy_input)
    # test out shape
    assert x3.shape == (2, 1, 10)
    # test outsize is correct
    assert shape == (1, 1)
    # test L = out_h * out_w
    assert shape[0] * shape[1] == x3.shape[1]

    # test the init_out_size with nn.Unfold
    assert patch_merge_3.init_out_size[1] == (input_size[0] - 2 * 4 - 1) // 2 + 1
    assert patch_merge_3.init_out_size[0] == (input_size[0] - 2 * 4 - 1) // 2 + 1
    _h = 11
    _w = 12
    input_size = (_h, _w)
    dummy_input = torch.rand(_b, _c, _h, _w)
    # test stride and norm
    patch_merge_3 = PatchEmbed(
        in_channels=_c,
        embed_dims=embed_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        normalization=nn.LayerNorm,
        input_size=input_size,
    )

    _, shape = patch_merge_3(dummy_input)
    # when input_size equal to real inputs
    # the out_size should be equal to `init_out_size`
    assert shape == patch_merge_3.init_out_size

    input_size = (_h, _w)
    dummy_input = torch.rand(_b, _c, _h, _w)
    # test stride and norm
    patch_merge_3 = PatchEmbed(
        in_channels=_c,
        embed_dims=embed_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        normalization=nn.LayerNorm,
        input_size=input_size,
    )

    _, shape = patch_merge_3(dummy_input)
    # when input_size equal to real inputs
    # the out_size should be equal to `init_out_size`
    assert shape == patch_merge_3.init_out_size

    # test adap padding
    for padding in ("same", "corner"):
        in_c = 2
        embed_dims = 3
        _b = 2

        # test stride is 1
        input_size = (5, 5)
        kernel_size = (5, 5)
        stride = (1, 1)
        dilation = 1
        bias = False

        x = torch.rand(_b, in_c, *input_size)
        patch_embed = PatchEmbed(
            in_channels=in_c,
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        x_out, out_size = patch_embed(x)
        assert x_out.size() == (_b, 25, 3)
        assert out_size == (5, 5)
        assert x_out.size(1) == out_size[0] * out_size[1]

        # test kernel_size == stride
        input_size = (5, 5)
        kernel_size = (5, 5)
        stride = (5, 5)
        dilation = 1
        bias = False

        x = torch.rand(_b, in_c, *input_size)
        patch_embed = PatchEmbed(
            in_channels=in_c,
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        x_out, out_size = patch_embed(x)
        assert x_out.size() == (_b, 1, 3)
        assert out_size == (1, 1)
        assert x_out.size(1) == out_size[0] * out_size[1]

        # test kernel_size == stride
        input_size = (6, 5)
        kernel_size = (5, 5)
        stride = (5, 5)
        dilation = 1
        bias = False

        x = torch.rand(_b, in_c, *input_size)
        patch_embed = PatchEmbed(
            in_channels=in_c,
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        x_out, out_size = patch_embed(x)
        assert x_out.size() == (_b, 2, 3)
        assert out_size == (2, 1)
        assert x_out.size(1) == out_size[0] * out_size[1]

        # test different kernel_size with different stride
        input_size = (6, 5)
        kernel_size = (6, 2)
        stride = (6, 2)
        dilation = 1
        bias = False

        x = torch.rand(_b, in_c, *input_size)
        patch_embed = PatchEmbed(
            in_channels=in_c,
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        x_out, out_size = patch_embed(x)
        assert x_out.size() == (_b, 3, 3)
        assert out_size == (1, 3)
        assert x_out.size(1) == out_size[0] * out_size[1]


def test_ffn():
    ffn = FFN(add_identity=True)

    input_tensor = torch.rand(2, 20, 256)
    input_tensor_nbc = input_tensor.transpose(0, 1)
    assert torch.allclose(ffn(input_tensor).sum(), ffn(input_tensor_nbc).sum())
    residual = torch.rand_like(input_tensor)

    torch.allclose(
        ffn(input_tensor, identity=residual).sum(),
        ffn(input_tensor).sum() + residual.sum() - input_tensor.sum(),
    )


def test_drop_path():
    drop_path = DropPath(drop_prob=0)
    test_in = torch.rand(2, 3, 4, 5)
    assert test_in is drop_path(test_in)

    drop_path = DropPath(drop_prob=0.1)
    drop_path.training = False
    test_in = torch.rand(2, 3, 4, 5)
    assert test_in is drop_path(test_in)
    drop_path.training = True
    assert test_in is not drop_path(test_in)
