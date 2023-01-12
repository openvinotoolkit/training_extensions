# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

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


@pytest.fixture(scope="session")
def tmp_dir_path():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(autouse=True)
def set_default_tmp_path(tmp_dir_path):
    origin_tmp_dir = os.environ.get("TMPDIR", None)
    os.environ["TMPDIR"] = str(tmp_dir_path)
    yield
    if origin_tmp_dir is None:
        del os.environ["TMPDIR"]
    else:
        os.environ["TMPDIR"] = origin_tmp_dir
