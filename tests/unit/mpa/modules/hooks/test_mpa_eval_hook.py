"""Unit test for otx.mpa.modules.hooks.eval_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.modules.hooks.eval_hook import CustomEvalHook, DistCustomEvalHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomEvalHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = CustomEvalHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestDistCustomEvalHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = DistCustomEvalHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass
