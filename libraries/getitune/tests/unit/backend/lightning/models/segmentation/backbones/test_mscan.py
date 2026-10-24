# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from getitune.backend.lightning.models.segmentation.backbones.mscan import DropPath, MSCANModule, drop_path


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_drop_path(dim: int):
    size = [10] + [2] * dim
    x = torch.ones(size)
    out = drop_path(x, 0.5, True)

    assert out.size() == x.size()
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_drop_path_not_train():
    x = torch.ones(2, 2, 2, 2)
    out = drop_path(x, 0.5, False)

    assert (x == out).all()
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_drop_path_zero_prob():
    x = torch.ones(2, 2, 2, 2)
    out = drop_path(x, 0.0, True)

    assert (x == out).all()
    assert out.dtype == x.dtype
    assert out.device == x.device


class TestDropPath:
    def test_init(self):
        drop_prob = 0.3
        drop_path = DropPath(drop_prob)

        assert drop_path.drop_prob == drop_prob

    def test_forward(self):
        drop_prob = 0.5
        drop_path = DropPath(drop_prob)
        drop_path.train()
        x = torch.ones(2, 2, 2, 2)

        out = drop_path.forward(x)

        assert out.size() == x.size()
        assert out.dtype == x.dtype
        assert out.device == x.device


class TestMSCABlock:
    def test_init(self):
        num_stages = 4
        mscan = MSCANModule(num_stages=num_stages)

        for i in range(num_stages):
            assert hasattr(mscan, f"patch_embed{i + 1}")
            assert hasattr(mscan, f"block{i + 1}")
            assert hasattr(mscan, f"norm{i + 1}")

    def test_forward(self):
        num_stages = 4
        mscan = MSCANModule(num_stages=num_stages)
        x = torch.rand(8, 3, 3, 3)
        out = mscan.forward(x)

        assert len(out) == num_stages
