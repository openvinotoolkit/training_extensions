# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Test of CSPDarknet.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/tests/test_models/test_backbones/test_csp_darknet.py
"""

import pytest
import torch
from otx.algo.detection.backbones.csp_darknet import CSPDarknet, Focus
from torch import nn
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/tests/test_models/test_backbones/utils.py#L26C1-L32C16
    """
    return all(not (isinstance(mod, _BatchNorm) and mod.training != train_state) for mod in modules)


def is_norm(modules):
    """Check if is one of the norms.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/tests/test_models/test_backbones/utils.py#L19-L23
    """
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


class TestFocus:
    def test_export(self) -> None:
        focus_model = Focus(3, 32)
        focus_model.requires_grad_(False)
        focus_model.cpu().eval()

        x = torch.rand(1, 3, 128, 128)

        results = focus_model.forward(x)

        assert results.shape == (1, 32, 64, 64)


class TestCSPDarknet:
    def test_init_with_large_frozen_stages(self) -> None:
        """Test __init__ with large frozen_stages."""
        with pytest.raises(ValueError):  # noqa: PT011
            # frozen_stages must in range(-1, len(arch_setting) + 1)
            CSPDarknet(frozen_stages=6)

    def test_init_with_large_out_indices(self) -> None:
        """Test __init__ with large out_indices."""
        with pytest.raises(AssertionError):
            CSPDarknet(out_indices=[6])

    def test_freeze_stages(self) -> None:
        """Test _freeze_stages."""
        frozen_stages = 1
        model = CSPDarknet(frozen_stages=frozen_stages)
        model.train()

        for mod in model.stem.modules():
            for param in mod.parameters():
                assert param.requires_grad is False
        for i in range(1, frozen_stages + 1):
            layer = getattr(model, f"stage{i}")
            for mod in layer.modules():
                if isinstance(mod, _BatchNorm):
                    assert mod.training is False
            for param in layer.parameters():
                assert param.requires_grad is False

    def test_train_with_norm_eval(self) -> None:
        """Test train with norm_eval=True."""
        model = CSPDarknet(norm_eval=True)
        model.train()

        assert check_norm_state(model.modules(), False)

    def test_forward(self) -> None:
        # Test CSPDarknet-P5 forward with widen_factor=0.5
        model = CSPDarknet(arch="P5", widen_factor=0.25, out_indices=range(5))
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        assert len(feat) == 5
        assert feat[0].shape == torch.Size((1, 16, 32, 32))
        assert feat[1].shape == torch.Size((1, 32, 16, 16))
        assert feat[2].shape == torch.Size((1, 64, 8, 8))
        assert feat[3].shape == torch.Size((1, 128, 4, 4))
        assert feat[4].shape == torch.Size((1, 256, 2, 2))

        # Test CSPDarknet-P6 forward with widen_factor=0.5
        model = CSPDarknet(arch="P6", widen_factor=0.25, out_indices=range(6), spp_kernal_sizes=(3, 5, 7))
        model.train()

        imgs = torch.randn(1, 3, 128, 128)
        feat = model(imgs)
        assert feat[0].shape == torch.Size((1, 16, 64, 64))
        assert feat[1].shape == torch.Size((1, 32, 32, 32))
        assert feat[2].shape == torch.Size((1, 64, 16, 16))
        assert feat[3].shape == torch.Size((1, 128, 8, 8))
        assert feat[4].shape == torch.Size((1, 192, 4, 4))
        assert feat[5].shape == torch.Size((1, 256, 2, 2))

        # Test CSPDarknet forward with dict(type='ReLU')
        model = CSPDarknet(widen_factor=0.125, activation=nn.ReLU, out_indices=range(5))
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        assert len(feat) == 5
        assert feat[0].shape == torch.Size((1, 8, 32, 32))
        assert feat[1].shape == torch.Size((1, 16, 16, 16))
        assert feat[2].shape == torch.Size((1, 32, 8, 8))
        assert feat[3].shape == torch.Size((1, 64, 4, 4))
        assert feat[4].shape == torch.Size((1, 128, 2, 2))

        # Test CSPDarknet with BatchNorm forward
        model = CSPDarknet(widen_factor=0.125, out_indices=range(5))
        for m in model.modules():
            if is_norm(m):
                assert isinstance(m, _BatchNorm)
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        assert len(feat) == 5
        assert feat[0].shape == torch.Size((1, 8, 32, 32))
        assert feat[1].shape == torch.Size((1, 16, 16, 16))
        assert feat[2].shape == torch.Size((1, 32, 8, 8))
        assert feat[3].shape == torch.Size((1, 64, 4, 4))
        assert feat[4].shape == torch.Size((1, 128, 2, 2))

        # Test CSPDarknet with custom arch forward
        arch_ovewrite = [[32, 56, 3, True, False], [56, 224, 2, True, False], [224, 512, 1, True, False]]
        model = CSPDarknet(arch_ovewrite=arch_ovewrite, widen_factor=0.25, out_indices=(0, 1, 2, 3))
        model.train()

        imgs = torch.randn(1, 3, 32, 32)
        feat = model(imgs)
        assert len(feat) == 4
        assert feat[0].shape == torch.Size((1, 8, 16, 16))
        assert feat[1].shape == torch.Size((1, 14, 8, 8))
        assert feat[2].shape == torch.Size((1, 56, 4, 4))
        assert feat[3].shape == torch.Size((1, 128, 2, 2))
