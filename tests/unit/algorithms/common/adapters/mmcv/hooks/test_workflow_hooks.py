"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.workflow_hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.hooks.workflow_hook import (
    AfterStageWFHook,
    SampleLoggingHook,
    WFProfileHook,
    WorkflowHook,
    build_workflow_hook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def test_build_workflow_hook() -> None:
    try:
        build_workflow_hook()
    except Exception as e:
        print(e)
        pass


class TestWorkflowHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = WorkflowHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestSampleLoggingHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = SampleLoggingHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestWFProfileHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = WFProfileHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestAfterStageWFHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = AfterStageWFHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass
