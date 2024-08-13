# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from otx.algo.visual_prompting.backbones.tiny_vit import (
    Attention,
    BasicLayer,
    Conv2d_BN,
    ConvLayer,
    MBConv,
    Mlp,
    PatchEmbed,
    PatchMerging,
    TinyViT,
    TinyViTBlock,
)
from torch import nn


class TestConv2d_BN:  # noqa: N801
    @pytest.fixture()
    def conv2d_bn(self) -> Conv2d_BN:
        return Conv2d_BN(a=1, b=1)

    def test_init(self, conv2d_bn) -> None:
        """Test __init__."""
        assert isinstance(conv2d_bn.c, nn.Conv2d)
        assert isinstance(conv2d_bn.bn, nn.BatchNorm2d)

    def test_fuse(self, conv2d_bn) -> None:
        """Test fuse."""
        fulsed_module = conv2d_bn.fuse()

        tmp_w = conv2d_bn.bn.weight / (conv2d_bn.bn.running_var + conv2d_bn.bn.eps) ** 0.5
        new_w = conv2d_bn.c.weight * tmp_w[:, None, None, None]
        new_b = conv2d_bn.bn.bias - conv2d_bn.bn.running_mean * tmp_w

        assert torch.isclose(fulsed_module.weight, new_w)
        assert torch.isclose(fulsed_module.bias, new_b)


class TestPatchEmbed:
    def test_forward(self) -> None:
        """Test forward."""
        patch_embed = PatchEmbed(in_chans=3, embed_dim=4, resolution=6, activation=nn.Identity)
        input_tensor = torch.rand(1, 3, 6, 6)

        results = patch_embed(input_tensor)

        assert results.shape == torch.Size((1, 4, 2, 2))


class TestMBConv:
    def test_forward(self) -> None:
        """Test forward."""
        mbconv = MBConv(in_chans=3, out_chans=3, expand_ratio=1.0, activation=nn.Identity, drop_path=1.0)
        input_tensor = torch.rand(1, 3, 24, 24)

        results = mbconv(input_tensor)

        assert results.shape == torch.Size((1, 3, 24, 24))


class TestPatchMerging:
    def test_forward(self) -> None:
        """Test forward."""
        patch_merging = PatchMerging(input_resolution=(6, 6), dim=3, out_dim=4, activation=nn.Identity)
        input_tensor = torch.rand(1, 3, 6, 6)

        results = patch_merging(input_tensor)

        assert results.shape == torch.Size((1, 9, 4))


class TestConvLayer:
    def test_forward(self) -> None:
        """Test forward."""
        conv_layer = ConvLayer(dim=3, input_resolution=(6, 6), depth=1, activation=nn.Identity)
        input_tensor = torch.rand(1, 3, 6, 6)

        results = conv_layer(input_tensor)

        assert results.shape == torch.Size((1, 3, 6, 6))


class TestMlp:
    def test_forward(self) -> None:
        """Test forward."""
        mlp = Mlp(in_features=4, hidden_features=5, out_features=6)
        input_tensor = torch.rand(1, 4)

        results = mlp(input_tensor)

        assert results.shape == torch.Size((1, 6))


class TestAttention:
    def test_forward(self) -> None:
        """Test forward."""
        attention = Attention(dim=4, key_dim=4, num_heads=1, attn_ratio=1, resolution=(2, 2))
        input_tensor = torch.rand(9, 4, 4)

        results = attention(input_tensor)

        assert results.shape == torch.Size((9, 4, 4))


class TestTinyViTBlock:
    @pytest.fixture()
    def tiny_vit_block(self) -> TinyViTBlock:
        return TinyViTBlock(
            dim=4,
            input_resolution=(6, 6),
            num_heads=1,
            window_size=2,
            mlp_ratio=1.0,
        )

    def test_forward(self, tiny_vit_block) -> None:
        """Test forward."""
        input_tensor = torch.rand(1, 36, 4)

        results = tiny_vit_block(input_tensor)

        assert results.shape == torch.Size((1, 36, 4))


class TestBasicLayer:
    @pytest.fixture()
    def basic_layer(self) -> BasicLayer:
        return BasicLayer(
            dim=4,
            input_resolution=(6, 6),
            depth=1,
            num_heads=1,
            window_size=2,
        )

    def test_forward(self, basic_layer) -> None:
        """Test forward."""
        input_tensor = torch.rand(1, 36, 4)

        results = basic_layer(input_tensor)

        assert results.shape == torch.Size((1, 36, 4))

    def test_extra_repr(self, basic_layer) -> None:
        """Test extra_repr."""
        assert basic_layer.extra_repr() == "dim=4, input_resolution=(6, 6), depth=1"


class TestTinyViT:
    @pytest.fixture()
    def tiny_vit(self) -> TinyViT:
        return TinyViT(
            img_size=1024,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            drop_path_rate=0.0,
            layer_lr_decay=2.0,
        )

    def test_forward(self, tiny_vit) -> None:
        """Test forward."""
        input_tensor = torch.rand(1, 3, 1024, 1024)

        results = tiny_vit(input_tensor)

        assert results.shape == torch.Size((1, 256, 64, 64))
