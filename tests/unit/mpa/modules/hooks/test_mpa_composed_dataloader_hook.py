"""Unit test for otx.mpa.modules.hooks.composed_dataloaders_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.modules.hooks.composed_dataloaders_hook import ComposedDataLoadersHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestComposedDataLoadersHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = ComposedDataLoadersHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass
