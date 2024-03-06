"""Test for otx.algorithms.common.adapters.mmcv.utils.fp16_utils"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from unittest.mock import MagicMock, patch
from torch import nn, torch
from otx.algorithms.common.adapters.mmcv.utils.fp16_utils import custom_auto_fp16
from otx.algorithms.common.adapters.mmcv.utils.fp16_utils import custom_force_fp32
from otx.algorithms.common.utils import is_xpu_available


@pytest.fixture
def test_module():
    class TestModule(torch.nn.Module):
        def __init__(self):
            super(TestModule, self).__init__()
            self.fp16_enabled = False

        @custom_auto_fp16()
        def test_method_fp16(self, arg1, arg2):
            return torch.tensor(arg1) + torch.tensor(arg2)

        @custom_auto_fp16(out_fp32=True)
        def test_method_force_out_fp32(self, arg1, arg2):
            return torch.tensor(arg1) + torch.tensor(arg2)

        @custom_force_fp32(out_fp16=False)
        def test_func_force_fp16_to_fp32(self, arg1, arg2):
            return torch.tensor(arg1) + torch.tensor(arg2)

        @custom_force_fp32(out_fp16=True)
        def test_func_force_fp32_out_fp16(self, arg1, arg2):
            return torch.tensor(arg1) + torch.tensor(arg2)

        def set_fp16(self, enabled):
            self.fp16_enabled = enabled

    return TestModule()


class TestCustomAutoFP16:
    def test_simple_apply(self, test_module):
        test_func = test_module.test_method_fp16
        # assertion simple ints
        assert test_func(5, 6) == 11
        # no fp16 enabled
        assert test_func(torch.tensor(5.3), torch.tensor(8.3)).dtype == torch.float32

    def test_fp16_enabled_true(self, test_module):
        test_module.set_fp16(enabled=True)
        test_func = test_module.test_method_fp16
        # check fp16 casting
        if not is_xpu_available():
            assert test_func(torch.tensor(5.3), torch.tensor(8.3)).dtype == torch.float16
        else:
            assert test_func(torch.tensor(5.3), torch.tensor(8.3)).dtype == torch.bfloat16

    def test_out_fp32_true(self, test_module):
        test_module.set_fp16(enabled=True)
        test_func = test_module.test_method_force_out_fp32
        # cast back to fp32
        assert test_func(torch.tensor(5.3), torch.tensor(8.3)).dtype == torch.float32


class TestCustomForceFP32:
    def test_simple_apply(self, test_module):
        test_func = test_module.test_func_force_fp16_to_fp32
        # assertion simple ints
        assert test_func(5, 6) == 11
        # no fp16 enabled
        assert test_func(torch.tensor(5.3), torch.tensor(8.3)).dtype == torch.float32

    @pytest.mark.skipif(is_xpu_available(), reason="cuda is not available")
    def test_fp16_enabled_true(self, test_module):
        test_module.set_fp16(enabled=True)
        test_func = test_module.test_func_force_fp16_to_fp32
        output_type = test_func(torch.tensor(5.3, dtype=torch.float16), torch.tensor(8.3, dtype=torch.float16)).dtype
        # check fp16 casting
        assert output_type == torch.float32

    def test_out_fp32_true(self, test_module):
        test_module.set_fp16(enabled=True)
        test_func = test_module.test_func_force_fp32_out_fp16
        output_type = test_func(torch.tensor(5.3, dtype=torch.float16), torch.tensor(8.3, dtype=torch.float16)).dtype
        # cast back to fp32
        assert output_type == torch.float16

    @pytest.mark.skipif(not is_xpu_available(), reason="XPU is not available")
    def test_fp16_enabled_xpu(self, test_module):
        # setup
        test_module.set_fp16(enabled=True)
        test_func = test_module.test_func_force_fp16_to_fp32
        output_type = test_func(torch.tensor(5.3, dtype=torch.bfloat16), torch.tensor(8.3, dtype=torch.bfloat16)).dtype
        # assertion
        assert output_type == torch.float32
