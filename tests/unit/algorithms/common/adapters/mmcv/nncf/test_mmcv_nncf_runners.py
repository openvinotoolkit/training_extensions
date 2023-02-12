# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmcv.nncf.test_helpers import (
    create_dataloader,
    create_nncf_runner,
)


class TestAccuracyAwareRunner:
    @e2e_pytest_unit
    def test_run(self):
        dataloader = create_dataloader()

        with tempfile.TemporaryDirectory() as tempdir:
            runner = create_nncf_runner(tempdir)
            runner.run([dataloader])
            assert [f for f in os.listdir(tempdir) if f.startswith("best_accuracy")]
