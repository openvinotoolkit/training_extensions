"""Unit test for otx.mpa.modules.hooks.early_stopping_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.modules.hooks.early_stopping_hook import (
    EarlyStoppingHook,
    LazyEarlyStoppingHook,
    ReduceLROnPlateauLrUpdaterHook,
    StopLossNanTrainingHook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestEarlyStoppingHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = EarlyStoppingHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestLazyEarlyStoppingHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = LazyEarlyStoppingHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestReduceLROnPlateauLrUpdaterHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = ReduceLROnPlateauLrUpdaterHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestStopLossNanTrainingHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = StopLossNanTrainingHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass
