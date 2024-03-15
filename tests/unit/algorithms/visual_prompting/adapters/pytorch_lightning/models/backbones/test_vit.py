"""Tests Vision Transformers."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import Tuple

import pytest
import torch
import torch.nn as nn

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.backbones.vit import (
    Attention,
    Block,
    PatchEmbed,
    ViT,
    add_decomposed_rel_pos,
    build_vit,
    get_rel_pos,
    window_partition,
    window_unpartition,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.utils import (
    MLPBlock,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestViT:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.embed_dim = 8
        self.num_heads = 2
        self.depth = 2
        self.vit = ViT(
            img_size=4, patch_size=2, embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads, out_chans=4
        )

    @e2e_pytest_unit
    def test_init(self):
        """Test init."""
        assert isinstance(self.vit.patch_embed, PatchEmbed)
        assert isinstance(self.vit.blocks, nn.ModuleList)
        assert len(self.vit.blocks) == self.depth
        assert isinstance(self.vit.neck, nn.Sequential)

    @e2e_pytest_unit
    @pytest.mark.parametrize("inputs,expected", [(torch.empty((1, 3, 4, 4)), (1, 4, 2, 2))])
    def test_forward(self, inputs: torch.Tensor, expected: Tuple[int]):
        """Test forward."""
        results = self.vit.forward(inputs)

        assert results.shape == expected


class TestBlock:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dim = 8
        self.num_heads = 2
        self.block = Block(dim=self.dim, num_heads=self.num_heads)

    @e2e_pytest_unit
    def test_init(self):
        """Test init."""
        assert isinstance(self.block.attn, Attention)
        assert isinstance(self.block.mlp, MLPBlock)

    @e2e_pytest_unit
    @pytest.mark.parametrize("inputs,expected", [(torch.empty((1, 4, 4, 8)), (1, 4, 4, 8))])
    def test_forward(self, inputs: torch.Tensor, expected: Tuple[int]):
        """Test forward."""
        results = self.block.forward(inputs)

        assert results.shape == expected


class TestAttention:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dim = 8
        self.num_heads = 2
        self.attention = Attention(dim=self.dim, num_heads=self.num_heads)

    @e2e_pytest_unit
    def test_init(self):
        """Test init."""
        assert isinstance(self.attention.qkv, nn.Linear)
        assert isinstance(self.attention.proj, nn.Linear)

    @e2e_pytest_unit
    @pytest.mark.parametrize("inputs,expected", [(torch.empty((1, 4, 4, 8)), (1, 4, 4, 8))])
    def test_forward(self, inputs: torch.Tensor, expected: Tuple[int]):
        """Test forward."""
        results = self.attention.forward(inputs)

        assert results.shape == expected


@e2e_pytest_unit
@pytest.mark.parametrize("inputs,window_size,expected", [(torch.empty((1, 4, 4, 4)), 2, ((4, 2, 2, 4), (4, 4)))])
def test_window_partition(inputs: torch.Tensor, window_size: int, expected: Tuple[int]):
    """Test window_partition."""
    results = window_partition(inputs, window_size)
    windows, (Hp, Wp) = results

    assert windows.shape == expected[0]
    assert (Hp, Wp) == expected[1]


@e2e_pytest_unit
@pytest.mark.parametrize(
    "windows,window_size,pad_hw,hw,expected", [(torch.empty((2, 2, 2, 2)), 2, (2, 2), (4, 4), (2, 2, 2, 2))]
)
def test_window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int], expected: Tuple[int]
):
    """Test window_unpartition."""
    results = window_unpartition(windows, window_size, pad_hw, hw)

    assert results.shape == expected


@e2e_pytest_unit
@pytest.mark.parametrize(
    "q_size,k_size,rel_pos,expected",
    [(2, 2, torch.empty((1, 2, 2)), (2, 2, 4)), (2, 2, torch.empty((3, 2, 2)), (2, 2, 2, 2))],
)
def test_get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor, expected: Tuple[int]):
    """Test get_rel_pos."""
    results = get_rel_pos(q_size, k_size, rel_pos)

    assert results.shape == expected


@e2e_pytest_unit
@pytest.mark.parametrize(
    "attn,q,rel_pos_h,rel_pos_w,q_size,k_size,expected",
    [
        (
            torch.empty((1, 4, 4)),
            torch.empty((1, 4, 1)),
            torch.empty((1, 2, 2, 2)),
            torch.empty((1, 2, 2, 2)),
            (2, 2),
            (2, 2),
            (1, 4, 4),
        )
    ],
)
def test_add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
    expected: Tuple[int],
):
    """Test add_decomposed_rel_pos."""
    results = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)

    assert results.shape == expected


class TestPatchEmbed:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.kernel_size = (2, 2)
        self.stride = (2, 2)
        self.embed_dim = 8
        self.patch_embed = PatchEmbed(kernel_size=self.kernel_size, stride=self.stride, embed_dim=self.embed_dim)

    @e2e_pytest_unit
    def test_init(self):
        """Test init."""
        assert isinstance(self.patch_embed.proj, nn.Conv2d)

    @e2e_pytest_unit
    @pytest.mark.parametrize("inputs,expected", [(torch.empty((8, 3, 2, 2)), (8, 1, 1, 8))])
    def test_forward(self, inputs: torch.Tensor, expected: Tuple[int]):
        """Test forward."""
        results = self.patch_embed.forward(torch.empty((8, 3, 2, 2)))

        assert results.shape == expected


@e2e_pytest_unit
@pytest.mark.parametrize("backbone", ["vit_b", "vit_l", "vit_h"])
def test_build_vit(mocker, backbone: str):
    """Test build_vit."""
    mocker_vit = mocker.patch("otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.backbones.vit.ViT")

    _ = build_vit(backbone=backbone, image_size=1024)

    mocker_vit.assert_called_once()
