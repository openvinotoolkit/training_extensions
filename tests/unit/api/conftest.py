# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.api.test_suite.pytest_insertions import *  # noqa #pylint: disable=unused-import
from tests.unit.api.fixtures.general import label_schema_example

pytest_plugins = get_pytest_plugins_from_otx()  # noqa: F405

otx_conftest_insertion(default_repository_name="otx/training_extensions/")  # noqa: F405


def pytest_addoption(parser):
    otx_pytest_addoption_insertion(parser)  # noqa: F405
