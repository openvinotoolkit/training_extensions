# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from tests.test_suite.e2e_test_system import e2e_pytest_unit

from otx.mpa.deploy.utils.operations_domain import add_domain, DOMAIN_CUSTOM_OPS_NAME


@e2e_pytest_unit
def test_add_domain():
    assert add_domain("abc") == DOMAIN_CUSTOM_OPS_NAME + "::" + "abc"
