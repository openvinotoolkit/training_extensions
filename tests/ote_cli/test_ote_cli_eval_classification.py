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
    args,
    wrong_paths,
    ote_eval_common
)

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()

templates = Registry('external').filter(task_type='ANOMALY_CLASSIFICATION').templates
templates_ids = [template.model_template_id for template in templates]


class TestEvalCommonAnomalyClassification:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_no_template(self, template):
        error_string = "the following arguments are required: template"
        ret = ote_eval_common(template, root, [])
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_no_test_files(self, template):
        error_string = "ote eval: error: the following arguments are required: --test-ann-files"
        command_args = [template.model_template_id,
                        '--test-data-roots',
                        f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--save-performance',
                        './trained_default_template/performance.json']
        ret = ote_eval_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_no_test_roots(self, template):
        error_string = "ote eval: error: the following arguments are required: --test-data-roots"
        command_args = [template.model_template_id,
                        '--test-ann-file',
                        f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--save-performance',
                        './trained_default_template/performance.json']
        ret = ote_eval_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_no_weights(self, template):
        error_string = "ote eval: error: the following arguments are required: --load-weights"
        command_args = [template.model_template_id,
                        '--test-ann-file',
                        f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                        '--test-data-roots',
                        f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                        '--save-performance',
                        './trained_default_template/performance.json']
        ret = ote_eval_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_wrong_paths_in_options(self, template):
        error_string = "Path is not valid"
        command_args = [template.model_template_id,
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
                temp = deepcopy(command_args)
                temp[i] = case
                ret = ote_eval_common(template, root, command_args)
                assert error_string in str(ret.stderr)
