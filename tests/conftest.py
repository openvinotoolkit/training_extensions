# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .test_suite.pytest_insertions import (
    get_pytest_plugins_from_otx,
    otx_conftest_insertion,
    otx_pytest_addoption_insertion,
)
from .unit.api.fixtures.general import label_schema_example  # noqa: F401

pytest_plugins = get_pytest_plugins_from_otx()  # noqa: F405

otx_conftest_insertion(default_repository_name="otx/training_extensions/")  # noqa: F405


def pytest_addoption(parser):
    otx_pytest_addoption_insertion(parser)  # noqa: F405
