"""Tests TinyViT."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import torch
import torch.nn as nn

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.backbones.tiny_vit import (
    Conv2d_BN,
    ConvLayer,
    DropPath,
    LayerNorm2d,
    TinyViT,
    MBConv,
    BasicLayer,
    TinyViTBlock,
    Mlp,
    Attention,
    PatchEmbed,
    PatchMerging,
    build_tiny_vit,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestConv2d_BN:
    def setup(self):
        self.conv2d_bn = Conv2d_BN(a=1, b=1)

    @e2e_pytest_unit
    def test_init(self):
        """Test __init__."""
        assert isinstance(self.conv2d_bn.c, nn.Conv2d)
        assert isinstance(self.conv2d_bn.bn, nn.BatchNorm2d)

    @e2e_pytest_unit
    def test_fuse(self):
        """Test fuse."""
        m = self.conv2d_bn.fuse()

        tmp_w = self.conv2d_bn.bn.weight / (self.conv2d_bn.bn.running_var + self.conv2d_bn.bn.eps) ** 0.5
        new_w = self.conv2d_bn.c.weight * tmp_w[:, None, None, None]
        new_b = self.conv2d_bn.bn.bias - self.conv2d_bn.bn.running_mean * tmp_w

        assert torch.isclose(m.weight, new_w)
        assert torch.isclose(m.bias, new_b)


class TestDropPath:
    @e2e_pytest_unit
    def test_repr(self, drop_prob: float = 1.0):
        """Test __repr__."""
        drop_path = DropPath(drop_prob=drop_prob)

        assert f"(drop_prob={drop_prob})" in repr(drop_path)


class TestPatchEmbed:
    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        patch_embed = PatchEmbed(in_chans=3, embed_dim=4, resolution=6, activation=nn.Identity)
        input_tensor = torch.rand(1, 3, 6, 6)

        results = patch_embed(input_tensor)

        assert results.shape == (1, 4, 2, 2)


class TestMBConv:
    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        mbconv = MBConv(in_chans=3, out_chans=3, expand_ratio=1.0, activation=nn.Identity, drop_path=1.0)
        input_tensor = torch.rand(1, 3, 24, 24)

        results = mbconv(input_tensor)

        assert results.shape == (1, 3, 24, 24)


class TestPatchMerging:
    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        patch_merging = PatchMerging(input_resolution=(6, 6), dim=3, out_dim=4, activation=nn.Identity)
        input_tensor = torch.rand(1, 3, 6, 6)

        results = patch_merging(input_tensor)

        assert results.shape == (1, 9, 4)


class TestConvLayer:
    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        conv_layer = ConvLayer(dim=3, input_resolution=(6, 6), depth=1, activation=nn.Identity)
        input_tensor = torch.rand(1, 3, 6, 6)

        results = conv_layer(input_tensor)

        assert results.shape == (1, 3, 6, 6)


class TestMlp:
    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        mlp = Mlp(in_features=4, hidden_features=5, out_features=6)
        input_tensor = torch.rand(1, 4)

        results = mlp(input_tensor)

        assert results.shape == (1, 6)


class TestAttention:
    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        attention = Attention(dim=4, key_dim=4, num_heads=1, attn_ratio=1, resolution=(2, 2))
        input_tensor = torch.rand(9, 4, 4)

        results = attention(input_tensor)

        assert results.shape == (9, 4, 4)


class TestTinyViTBlock:
    def setup(self):
        self.dim = 4
        self.input_resolution = (6, 6)
        self.num_heads = 1
        self.window_size = 2
        self.mlp_ratio = 1.0
        self.tiny_vit_block = TinyViTBlock(
            dim=self.dim,
            input_resolution=self.input_resolution,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
        )

    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        input_tensor = torch.rand(1, 36, 4)

        results = self.tiny_vit_block(input_tensor)

        assert results.shape == (1, 36, 4)

    @e2e_pytest_unit
    def test_extra_repr(self):
        """Test extra_repr."""
        (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        ) == self.tiny_vit_block.extra_repr()


class TestBasicLayer:
    def setup(self):
        self.dim = 4
        self.input_resolution = (6, 6)
        self.depth = 1
        self.basic_layer = BasicLayer(
            dim=self.dim, input_resolution=self.input_resolution, depth=self.depth, num_heads=1, window_size=2
        )

    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        input_tensor = torch.rand(1, 36, 4)

        results = self.basic_layer(input_tensor)

        assert results.shape == (1, 36, 4)

    @e2e_pytest_unit
    def test_extra_repr(self):
        """Test extra_repr."""
        assert (
            f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
            == self.basic_layer.extra_repr()
        )


class TestLayerNorm2d:
    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        layer_norm_2d = LayerNorm2d(num_channels=4)
        input_tensor = torch.rand(1, 4, 4, 4)

        results = layer_norm_2d(input_tensor)

        assert results.shape == (1, 4, 4, 4)


class TestTinyViT:
    def setup(self):
        self.tiny_vit = TinyViT(
            img_size=1024,
            num_classes=1,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            drop_path_rate=0.0,
            layer_lr_decay=2.0,
        )

    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        input_tensor = torch.rand(1, 3, 1024, 1024)

        results = self.tiny_vit(input_tensor)

        assert results.shape == (1, 256, 64, 64)


@e2e_pytest_unit
def test_build_tiny_vit(mocker):
    """Test build_tiny_vit."""
    mocker_vit = mocker.patch(
        "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.backbones.tiny_vit.TinyViT"
    )

    _ = build_tiny_vit(img_size=1024)

    mocker_vit.assert_called_once()
