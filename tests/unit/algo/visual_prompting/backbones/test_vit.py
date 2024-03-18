# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from otx.algo.visual_prompting.backbones.vit import (
    Attention,
    Block,
    PatchEmbed,
    ViT,
    add_decomposed_rel_pos,
    get_rel_pos,
    window_partition,
    window_unpartition,
)
from otx.algo.visual_prompting.utils import MLPBlock
from torch import Tensor, nn


class TestViT:
    @pytest.fixture()
    def vit(self) -> ViT:
        return ViT(img_size=4, patch_size=2, embed_dim=8, depth=2, num_heads=2, out_chans=4)

    def test_init(self, vit) -> None:
        """Test init."""
        assert isinstance(vit.patch_embed, PatchEmbed)
        assert isinstance(vit.blocks, nn.ModuleList)
        assert len(vit.blocks) == 2
        assert isinstance(vit.neck, nn.Sequential)

    @pytest.mark.parametrize(("inputs", "expected"), [(torch.empty((1, 3, 4, 4)), (1, 4, 2, 2))])
    def test_forward(self, vit, inputs: Tensor, expected: tuple[int]) -> None:
        """Test forward."""
        results = vit.forward(inputs)

        assert results.shape == expected


class TestBlock:
    @pytest.fixture()
    def block(self) -> Block:
        return Block(dim=8, num_heads=2)

    def test_init(self, block) -> None:
        """Test init."""
        assert isinstance(block.attn, Attention)
        assert isinstance(block.mlp, MLPBlock)

    @pytest.mark.parametrize(("inputs", "expected"), [(torch.empty((1, 4, 4, 8)), (1, 4, 4, 8))])
    def test_forward(self, block, inputs: Tensor, expected: tuple[int]) -> None:
        """Test forward."""
        results = block.forward(inputs)

        assert results.shape == expected


class TestAttention:
    @pytest.fixture()
    def attention(self) -> Attention:
        return Attention(dim=8, num_heads=2)

    def test_init(self, attention) -> None:
        """Test init."""
        assert isinstance(attention.qkv, nn.Linear)
        assert isinstance(attention.proj, nn.Linear)

    @pytest.mark.parametrize(("inputs", "expected"), [(torch.empty((1, 4, 4, 8)), (1, 4, 4, 8))])
    def test_forward(self, attention, inputs: Tensor, expected: tuple[int]) -> None:
        """Test forward."""
        results = attention.forward(inputs)

        assert results.shape == expected


@pytest.mark.parametrize(
    ("inputs", "window_size", "expected"),
    [(torch.empty((1, 4, 4, 4)), 2, ((4, 2, 2, 4), (4, 4)))],
)
def test_window_partition(inputs: Tensor, window_size: int, expected: tuple[int]) -> None:
    """Test window_partition."""
    results = window_partition(inputs, window_size)
    windows, (hp, wp) = results

    assert windows.shape == expected[0]
    assert (hp, wp) == expected[1]


@pytest.mark.parametrize(
    ("windows", "window_size", "pad_hw", "hw", "expected"),
    [(torch.empty((2, 2, 2, 2)), 2, (2, 2), (4, 4), (2, 2, 2, 2))],
)
def test_window_unpartition(
    windows: Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
    expected: tuple[int],
) -> None:
    """Test window_unpartition."""
    results = window_unpartition(windows, window_size, pad_hw, hw)

    assert results.shape == expected


@pytest.mark.parametrize(
    ("q_size", "k_size", "rel_pos", "expected"),
    [(2, 2, torch.empty((1, 2, 2)), (2, 2, 4)), (2, 2, torch.empty((3, 2, 2)), (2, 2, 2, 2))],
)
def test_get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor, expected: tuple[int]) -> None:
    """Test get_rel_pos."""
    results = get_rel_pos(q_size, k_size, rel_pos)

    assert results.shape == expected


@pytest.mark.parametrize(
    ("attn", "q", "rel_pos_h", "rel_pos_w", "q_size", "k_size", "expected"),
    [
        (
            torch.empty((1, 4, 4)),
            torch.empty((1, 4, 1)),
            torch.empty((1, 2, 2, 2)),
            torch.empty((1, 2, 2, 2)),
            (2, 2),
            (2, 2),
            (1, 4, 4),
        ),
    ],
)
def test_add_decomposed_rel_pos(
    attn: Tensor,
    q: Tensor,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
    expected: tuple[int],
) -> None:
    """Test add_decomposed_rel_pos."""
    results = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)

    assert results.shape == expected


class TestPatchEmbed:
    @pytest.fixture()
    def patch_embed(self) -> PatchEmbed:
        return PatchEmbed(kernel_size=(2, 2), stride=(2, 2), embed_dim=8)

    def test_init(self, patch_embed) -> None:
        """Test init."""
        assert isinstance(patch_embed.proj, nn.Conv2d)

    @pytest.mark.parametrize(("inputs", "expected"), [(torch.empty((8, 3, 2, 2)), (8, 1, 1, 8))])
    def test_forward(self, patch_embed, inputs: Tensor, expected: tuple[int]) -> None:
        """Test forward."""
        results = patch_embed.forward(torch.empty((8, 3, 2, 2)))

        assert results.shape == expected
