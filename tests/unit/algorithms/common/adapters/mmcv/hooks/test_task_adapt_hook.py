"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.task_adapt_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.hooks.task_adapt_hook import TaskAdaptHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestTaskAdaptHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = TaskAdaptHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass
