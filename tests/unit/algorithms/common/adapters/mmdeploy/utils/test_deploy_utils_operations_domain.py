# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmdeploy.utils.operations_domain import (
    DOMAIN_CUSTOM_OPS_NAME,
    add_domain,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_add_domain():
    assert add_domain("abc") == DOMAIN_CUSTOM_OPS_NAME + "::" + "abc"
