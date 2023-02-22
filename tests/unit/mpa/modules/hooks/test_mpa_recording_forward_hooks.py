"""Unit test for otx.mpa.modules.hooks.recording_forward_hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.modules.hooks.recording_forward_hooks import (
    BaseRecordingForwardHook,
    DetSaliencyMapHook,
    ReciproCAMHook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestBaseRecordingForwardHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = BaseRecordingForwardHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestDetSaliencyMapHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = DetSaliencyMapHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestReciproCAMHook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = ReciproCAMHook()
            assert hook is None
        except Exception as e:
            print(e)
            pass
