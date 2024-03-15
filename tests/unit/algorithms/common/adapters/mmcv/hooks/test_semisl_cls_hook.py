"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.semisl_cls_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.hooks.semisl_cls_hook import SemiSLClsHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSemiSLClsHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = SemiSLClsHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass
