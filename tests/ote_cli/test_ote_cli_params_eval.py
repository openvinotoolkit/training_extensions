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
    wrong_paths,
    args,
)

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()


class OTECliParamsEval:
    @pytest.fixture()
    def templates(self, algo_be):
        return Registry('external').filter(task_type=algo_be).templates

    @e2e_pytest_component
    def test_create_venv(self, templates):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    def test_ote_eval_no_test_files(self, templates):
        error_string = "ote eval: error: the following arguments are required: --test-ann-files"
        for template in templates:
            command_line = ['ote',
                            'eval',
                            template.model_template_id,
                            '--test-data-roots',
                            f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                            '--load-weights',
                            './trained_default_template/weights.pth',
                            '--save-performance',
                            './trained_default_template/performance.json']
            assert error_string in str(run(command_line, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_eval_no_test_roots(self, templates):
        error_string = "ote eval: error: the following arguments are required: --test-data-roots"
        for template in templates:
            command_line = ['ote',
                            'eval',
                            template.model_template_id,
                            '--test-ann-file',
                            f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                            '--load-weights',
                            './trained_default_template/weights.pth',
                            '--save-performance',
                            './trained_default_template/performance.json']
            assert error_string in str(run(command_line, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_eval_no_weights(self, templates):
        error_string = "ote eval: error: the following arguments are required: --load-weights"
        for template in templates:
            command_line = ['ote',
                            'eval',
                            template.model_template_id,
                            '--test-ann-file',
                            f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                            '--test-data-roots',
                            f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                            '--save-performance',
                            './trained_default_template/performance.json']
            assert error_string in str(run(command_line, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_eval_wrong_paths_in_options(self, templates):
        for template in templates:
            command_line = ['ote',
                            'eval',
                            template.model_template_id,
                            '--test-ann-file',
                            f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                            '--test-data-roots',
                            f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                            '--load-weights',
                            './trained_default_template/weights.pth',
                            '--save-performance',
                            './trained_default_template/performance.json']
            for i in [4, 6, 8]:
                for case in wrong_paths.values():
                    temp = deepcopy(command_line)
                    temp[i] = case
                    assert "Path is not valid" in str(run(temp, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_eval_no_template(self):
        error_string = "ote eval: error: the following arguments are required: template"
        command_line = ['ote',
                        'eval']
        assert error_string in str(run(command_line, capture_output=True).stderr)
