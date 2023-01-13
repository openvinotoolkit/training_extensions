# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import pytest

from ote_sdk.test_suite.pytest_insertions import *  # noqa #pylint: disable=unused-import

pytest_plugins = get_pytest_plugins_from_ote()

ote_conftest_insertion(default_repository_name="ote/training_extensions/")


def pytest_addoption(parser):
    ote_pytest_addoption_insertion(parser)


@pytest.fixture(autouse=True, scope="session")
def manage_tm_config_for_testing():
    # check file existance both 'isip' and 'openvino_telemetry' if not, create it.
    # and backup contents if exist
    cfg_dir = os.path.join(os.path.expanduser("~"), "intel")
    isip_path = os.path.join(cfg_dir, "isip")
    otm_path = os.path.join(cfg_dir, "openvino_telemetry")
    isip_exist = os.path.exists(isip_path)
    otm_exist = os.path.exists(otm_path)

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
