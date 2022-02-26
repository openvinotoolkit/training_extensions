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

import os
from subprocess import run

import pytest
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component

from ote_cli.registry import Registry

from common import (
    collect_env_vars,
    create_venv,
    get_some_vars,
    ote_demo_deployment_testing,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_deployment_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_hpo_testing,
    ote_train_testing,
    ote_export_testing,
    pot_optimize_testing,
    pot_eval_testing,
    nncf_optimize_testing,
    nncf_export_testing,
    nncf_eval_testing,
    nncf_eval_openvino_testing,
    wrong_paths,
    args,
)

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()

valid_paths = {'same_folder': '.',
               'upper_folder': '..',
               'external': 'external'
               }


class OTECliParamsFind:
    @pytest.fixture()
    def task_type(self, algo_be):
        return algo_be

    @pytest.fixture()
    def templates(self, algo_be):
        return Registry('external').filter(task_type=algo_be).templates

    @e2e_pytest_component
    def test_create_venv(self, templates):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("path", valid_paths.values(), ids=valid_paths.keys())
    def test_ote_cli_find_root(self, path):
        cmd = ['ote',
               'find',
               '--root',
               path
               ]
        assert run(cmd).returncode == 0

    @e2e_pytest_component
    def test_ote_cli_task_type(self, task_type):
        cmd = ['ote',
               'find',
               '--task_type',
               task_type
               ]
        assert run(cmd).returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("path", wrong_paths.values(), ids=wrong_paths.keys())
    def test_ote_cli_find_root_wrong_path(self, path):
        cmd = ['ote',
               'find'
               '--root',
               path
               ]
        assert run(cmd).returncode != 0

    @e2e_pytest_component
    def test_ote_cli_find_task_type_not_set(self):
        cmd = ['ote',
               'find'
               '--task_id',
               ]
        assert run(cmd).returncode != 0

    @e2e_pytest_component
    def test_ote_cli_find(self):
        cmd = ['ote',
               'find'
               ]
        assert run(cmd).returncode == 0

    @e2e_pytest_component
    def test_ote_cli_find_help(self):
        cmd = ['ote',
               'find'
               ]
        assert run(cmd + ['-h']).returncode == 0
        assert run(cmd + ['--help']).returncode == 0
