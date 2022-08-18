# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.api.test_suite.pytest_insertions import *  # noqa #pylint: disable=unused-import
from otx.api.tests.fixtures.general import (  # noqa #pylint: disable=unused-import
    label_schema_example,
)

pytest_plugins = get_pytest_plugins_from_ote()  # noqa: F405

ote_conftest_insertion(default_repository_name="ote/training_extensions/")  # noqa: F405


def pytest_addoption(parser):
    ote_pytest_addoption_insertion(parser)  # noqa: F405
