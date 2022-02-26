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
from copy import deepcopy

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
    wrong_paths
)

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()


class OTECliParamsDeploy:
    @pytest.fixture()
    def templates(self, algo_be):
        return Registry('external').filter(task_type=algo_be).templates

    @e2e_pytest_component
    def test_create_venv(self, templates):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    def test_ote_deploy_no_weights(self, templates):
        expected_error = "ote deploy: error: the following arguments are required: --load-weights"
        for template in templates:
            deployment_dir = f'./deployed_{template.model_template_id}'
            command_line = ['ote',
                            'deploy',
                            template.model_template_id,
                            f'--save-model-to',
                            deployment_dir]
            assert expected_error in str(run(command_line, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_deploy_wrong_paths(self, templates):
        for template in templates:
            command_line = ['ote',
                            'deploy',
                            template.model_template_id,
                            '--load-weights',
                            'WRONG/PATH.bin',
                            '--save-model-to',
                            '']
            for case in wrong_paths.values():
                temp = deepcopy(command_line)
                temp[4] = case
                assert "Path is not valid" in str(run(temp, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_deploy_no_template(self):
        error_string = "ote deploy: error: the following arguments are required: template, --load-weights"
        command_line = ['ote',
                        'deploy']
        assert error_string in str(run(command_line, capture_output=True).stderr)
