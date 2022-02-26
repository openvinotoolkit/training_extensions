"""Tests for input parameters with OTE CLI"""

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

import pytest
from subprocess import run

from common import wrong_paths


@pytest.fixture()
def task_type(algo_be):
    return algo_be


valid_paths = {'same_folder': '.',
               'upper_folder': '..',
               'external': 'external'
               }


@pytest.mark.parametrize("path", valid_paths.values(), ids=valid_paths.keys())
def test_ote_cli_find_root(path):
    cmd = ['ote',
           'find',
           '--root',
           path
           ]
    assert run(cmd).returncode == 0


def test_ote_cli_task_type(task_type):
    cmd = ['ote',
           'find',
           '--task_type',
           task_type
           ]
    assert run(cmd).returncode == 0


@pytest.mark.parametrize("path", wrong_paths.values(), ids=wrong_paths.keys())
def test_ote_cli_find_root_wrong_path(path):
    cmd = ['ote',
           'find'
           '--root',
           path
           ]
    assert run(cmd).returncode != 0


def test_ote_cli_find_task_type_not_set():
    cmd = ['ote',
           'find'
           '--task_id',
           ]
    assert run(cmd).returncode != 0


def test_ote_cli_find():
    cmd = ['ote',
           'find'
           ]
    assert run(cmd).returncode == 0


def test_ote_cli_find_help():
    cmd = ['ote',
           'find'
           ]
    assert run(cmd + ['-h']).returncode == 0
    assert run(cmd + ['--help']).returncode == 0
