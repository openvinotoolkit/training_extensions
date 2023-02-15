# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.utils import get_logger

from otx.algorithms.common.adapters.mmcv.nncf.hooks import CompressionHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmcv.nncf.test_helpers import (
    create_nncf_model,
)


class TestCompressionHook:
    @e2e_pytest_unit
    def test_after_train_iter(self):
        class SimpleRunner:
            def __init__(self):
                self.logger = get_logger("mmcv")
                self.rank = 0

        runner = SimpleRunner()
        ctrl, _ = create_nncf_model()
        compression_hook = CompressionHook(ctrl)
        compression_hook.after_train_iter(runner)

    @e2e_pytest_unit
    def test_after_train_epoch(self):
        class SimpleRunner:
            def __init__(self):
                self.logger = get_logger("mmcv")
                self.rank = 0

        runner = SimpleRunner()
        ctrl, _ = create_nncf_model()
        compression_hook = CompressionHook(ctrl)
        compression_hook.after_train_epoch(runner)

    @e2e_pytest_unit
    def test_before_run(self):
        class SimpleRunner:
            def __init__(self):
                self.logger = get_logger("mmcv")
                self.rank = 0

        runner = SimpleRunner()
        ctrl, _ = create_nncf_model()
        compression_hook = CompressionHook(ctrl)
        compression_hook.before_run(runner)
