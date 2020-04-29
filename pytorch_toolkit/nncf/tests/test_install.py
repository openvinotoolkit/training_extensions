"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import subprocess
import sys
import pytest
import os


def test_install(install_type, tmp_path):
    if install_type is None:
        pytest.skip("Please specify type of installation")
    tests_dir = os.path.dirname(__file__)
    cur_dir = os.path.dirname(tests_dir)
    install_path = str(tmp_path.joinpath("install"))
    if sys.version_info[:2] == (3, 5):
        subprocess.call("virtualenv -ppython3.5 {}".format(install_path), shell=True)
    else:
        subprocess.call("{} -m venv {}".format(sys.executable, install_path), shell=True)
    python_executable_with_venv = str(". {0}/bin/activate && {0}/bin/python".format(install_path))
    if install_type == "CPU":
        subprocess.run(
            "{} {}/setup.py develop --cpu-only".format(python_executable_with_venv, cur_dir), check=True, shell=True)
        subprocess.run(
            "{} {}/install_checks.py cpu".format(python_executable_with_venv, tests_dir), check=True, shell=True)
    else:
        subprocess.run(
            "{} {}/setup.py develop".format(python_executable_with_venv, cur_dir), check=True, shell=True)
        subprocess.run(
            "{} {}/install_checks.py cuda".format(python_executable_with_venv, tests_dir), check=True, shell=True)
