# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.algorithms.classification.adapters.mmcls.models.necks import MMOVNeck
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.ov.models.mmcls.test_helpers import create_ov_model


class TestMMOVNeck:
    @pytest.fixture(autouse=True)
    def setup(self):
        ov_model = create_ov_model()
        self.model = MMOVNeck(
            model_path_or_model=ov_model,
        )

    @e2e_pytest_unit
    def test_parser(self):
        pass
