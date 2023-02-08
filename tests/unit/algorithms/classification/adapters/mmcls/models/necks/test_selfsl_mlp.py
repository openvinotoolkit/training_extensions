# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict

import pytest
import torch
import torch.nn as nn

from otx.algorithms.classification.adapters.mmcls import SelfSLMLP
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSelfSLMLP:
    @e2e_pytest_unit
    @pytest.mark.parametrize("use_conv", [True, False])
    @pytest.mark.parametrize("with_avg_pool", [True, False])
    def test_init(self, use_conv: bool, with_avg_pool: bool) -> None:
        """Test __init__ function."""
        selfslmlp = SelfSLMLP(
            in_channels=2, hid_channels=2, out_channels=2, use_conv=use_conv, with_avg_pool=with_avg_pool
        )

        if with_avg_pool:
            assert isinstance(selfslmlp.avgpool, nn.AdaptiveAvgPool2d)

        if use_conv:
            assert isinstance(selfslmlp.mlp[0], nn.Conv2d)
            assert isinstance(selfslmlp.mlp[3], nn.Conv2d)
        else:
            assert isinstance(selfslmlp.mlp[0], nn.Linear)
            assert isinstance(selfslmlp.mlp[3], nn.Linear)

    @e2e_pytest_unit
    @pytest.mark.parametrize("init_linear", ["normal", "kaiming"])
    @pytest.mark.parametrize("std", [0.01, 1.0, 10.0])
    @pytest.mark.parametrize("bias", [0.1, 0.0, 1.0])
    def test_init_weights(self, init_linear: str, std: float, bias: float) -> None:
        """Test init_weights function.

        Check if weights of nn.Linear was changed except for biases.
        BatchNorm weights are already set to 1, so it isn't required to be checked.
        """

        def gather_weight_mean_std(modules):
            mean = []
            std = []
            for module in modules:
                if isinstance(module, nn.Linear):
                    mean.append(module.weight.mean())
                    std.append(module.weight.std())
            return mean, std

        selfslmlp = SelfSLMLP(in_channels=2, hid_channels=2, out_channels=2, use_conv=False, with_avg_pool=True)

        orig_mean, orig_std = gather_weight_mean_std(selfslmlp.modules())

        selfslmlp.init_weights(init_linear, std, bias)

        updated_mean, updated_std = gather_weight_mean_std(selfslmlp.modules())

        for origs, updateds in zip([orig_mean, orig_std], [updated_mean, updated_std]):
            for orig, updated in zip(origs, updateds):
                assert orig != updated

    @e2e_pytest_unit
    @pytest.mark.parametrize("init_linear", ["undefined"])
    def test_init_weights_undefined_initialization(self, init_linear: str) -> None:
        """Test init_weights function when undefined initialization is given."""
        selfslmlp = SelfSLMLP(in_channels=2, hid_channels=2, out_channels=2, use_conv=False, with_avg_pool=True)

        with pytest.raises(ValueError):
            selfslmlp.init_weights(init_linear)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "inputs,norm_cfg,use_conv,with_avg_pool,expected",
        [
            (torch.rand((2, 2)), dict(type="BN1d"), False, False, torch.Size([2, 2])),
            (torch.rand((2, 2, 2, 2)), dict(type="BN1d"), False, True, torch.Size([2, 2])),
            (torch.rand((2, 2, 2, 2)), dict(type="BN2d"), True, False, torch.Size([2, 2, 2, 2])),
            (torch.rand((2, 2, 2, 2)), dict(type="BN2d"), True, True, torch.Size([2, 2, 1, 1])),
        ],
    )
    def test_forward_tensor(
        self, inputs: torch.Tensor, norm_cfg: Dict, use_conv: bool, with_avg_pool: bool, expected: torch.Size
    ) -> None:
        """Test forward function for tensor."""
        selfslmlp = SelfSLMLP(
            in_channels=2,
            hid_channels=2,
            out_channels=2,
            norm_cfg=norm_cfg,
            use_conv=use_conv,
            with_avg_pool=with_avg_pool,
        )

        results = selfslmlp(inputs)

        assert results.shape == expected

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "inputs,norm_cfg,use_conv,with_avg_pool,expected",
        [
            ([torch.rand((2, 2)), torch.rand((2, 2))], dict(type="BN1d"), False, False, torch.Size([2, 2])),
            ([torch.rand((2, 2, 2, 2)), torch.rand((2, 2, 2, 2))], dict(type="BN1d"), False, True, torch.Size([2, 2])),
            (
                [torch.rand((2, 2, 2, 2)), torch.rand((2, 2, 2, 2))],
                dict(type="BN2d"),
                True,
                False,
                torch.Size([2, 2, 2, 2]),
            ),
            (
                [torch.rand((2, 2, 2, 2)), torch.rand((2, 2, 2, 2))],
                dict(type="BN2d"),
                True,
                True,
                torch.Size([2, 2, 1, 1]),
            ),
        ],
    )
    def test_forward_list_tuple(
        self, inputs: torch.Tensor, norm_cfg: Dict, use_conv: bool, with_avg_pool: bool, expected: torch.Size
    ) -> None:
        """Test forward function for list or tuple."""
        selfslmlp = SelfSLMLP(
            in_channels=2,
            hid_channels=2,
            out_channels=2,
            norm_cfg=norm_cfg,
            use_conv=use_conv,
            with_avg_pool=with_avg_pool,
        )

        results = selfslmlp(inputs)

        assert results.shape == expected

    @e2e_pytest_unit
    @pytest.mark.parametrize("inputs", ["unsupported", 1])
    def test_forward_unsupported_format(self, inputs: str) -> None:
        """Test forward function for unsupported format."""
        selfslmlp = SelfSLMLP(
            in_channels=2,
            hid_channels=2,
            out_channels=2,
        )

        with pytest.raises(TypeError):
            selfslmlp(inputs)
