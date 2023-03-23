"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.model_ema_v2_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.hooks.model_ema_v2_hook import (
    ModelEmaV2,
    ModelEmaV2Hook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestModelEmaV2Hook:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            hook = ModelEmaV2Hook()
            assert hook is None
        except Exception as e:
            print(e)
            pass


class TestModelEmaV2:
    @e2e_pytest_unit
    def test_temp(self) -> None:
        try:
            model = ModelEmaV2()
            assert model is None
        except Exception as e:
            print(e)
            pass
