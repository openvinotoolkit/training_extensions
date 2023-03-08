# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

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
def tmp_dir_path(request) -> Generator[Path, None, None]:
    prefix = request.config.getoption("--test-work-dir")
    with TemporaryDirectory(prefix=prefix) as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(autouse=True)
def set_default_tmp_path(tmp_dir_path: Path) -> Generator[None, None, None]:
    origin_tmp_dir = os.environ.get("TMPDIR", None)
    os.environ["TMPDIR"] = str(tmp_dir_path)
    yield
    if origin_tmp_dir is None:
        del os.environ["TMPDIR"]
    else:
        os.environ["TMPDIR"] = origin_tmp_dir


@pytest.fixture(autouse=True, scope="session")
def manage_tm_config_for_testing():
    # check file existance both 'isip' and 'openvino_telemetry' if not, create it.
    # and backup contents if exist
    cfg_dir = os.path.join(os.path.expanduser("~"), "intel")
    isip_path = os.path.join(cfg_dir, "isip")
    otm_path = os.path.join(cfg_dir, "openvino_telemetry")
    isip_exist = os.path.exists(isip_path)
    otm_exist = os.path.exists(otm_path)

    created_cfg_dir = False

    if not os.path.exists(cfg_dir):
        created_cfg_dir = True
        os.makedirs(cfg_dir)

    isip_backup = None

    if not isip_exist:
        with open(isip_path, "w") as f:
            f.write("0")
    else:
        with open(isip_path, "r") as f:
            isip_backup = f.read()

    otm_backup = None
    if not otm_exist:
        with open(otm_path, "w") as f:
            f.write("0")
    else:
        with open(otm_path, "r") as f:
            otm_backup = f.read()

    yield

    # restore or remove
    if not isip_exist:
        os.remove(isip_path)
    else:
        if isip_backup is not None:
            with open(isip_path, "w") as f:
                f.write(isip_backup)

    if not otm_exist:
        os.remove(otm_path)
    else:
        if otm_backup is not None:
            with open(otm_path, "w") as f:
                f.write(otm_backup)

    if created_cfg_dir:
        os.rmdir(cfg_dir)
