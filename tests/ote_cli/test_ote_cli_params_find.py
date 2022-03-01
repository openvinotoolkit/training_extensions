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

from subprocess import run
from copy import deepcopy

import os
import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_cli.registry import Registry

from common import (
    create_venv,
    get_some_vars,
    ote_demo_deployment_testing,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_deployment_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_train_testing,
    ote_export_testing,
    pot_optimize_testing,
    pot_eval_testing,
    nncf_optimize_testing,
    nncf_export_testing,
    nncf_eval_testing,
    nncf_eval_openvino_testing,
    wrong_paths,
    ote_find_common
)

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()

templates = Registry('external').templates
templates_ids = [template.model_template_id for template in templates]


class TestOTECliFindParams:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_cli_find(self, template):
        ret = ote_find_common(template, [])
        assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_cli_find_root(self, template):
        valid_paths = {'same_folder': '.',
                       'upper_folder': '..',
                       'external': 'external'
                       }
        for path in valid_paths.values():
            cmd = ['--root', path]
            ret = ote_find_common(template, cmd)
            assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_cli_task_type(self, template):
        task_types = ["ANOMALY_CLASSIFICATION", "CLASSIFICATION", "DETECTION", "SEGMENTATION"]
        for task_type in task_types:
            cmd = ['--task_type', task_type]
            ret = ote_find_common(template, cmd)
            assert ret.returncode == 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_cli_find_root_wrong_path(self, template):
        for path in wrong_paths.values():
            cmd = ['--root', path]
            ret = ote_find_common(template, cmd)
            assert ret.returncode != 0

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_cli_find_task_type_not_set(self, template):
        cmd = ['--task_id', '']
        ret = ote_find_common(template, cmd)
        assert ret.returncode != 0
