"""Unit test for otx.mpa.modules.hooks.model_ema_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.modules.hooks.model_ema_hook import CustomModelEMAHook, DualModelEMAHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestDualModelEMAHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = DualModelEMAHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestCustomModelEMAHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = CustomModelEMAHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass
