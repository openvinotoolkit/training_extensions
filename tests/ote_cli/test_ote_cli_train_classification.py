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
    ote_train_common
)

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()

templates = Registry('external').filter(task_type='CLASSIFICATION').templates
templates_ids = [template.model_template_id for template in templates]


class TestTrainCommonClassification:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_template(self, template):
        error_string = "ote train: error: the following arguments are required: template"
        ret = ote_train_common(template, root, [])
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_train_ann_file(self, template):
        error_string = "ote train: error: the following arguments are required: --train-ann-files"
        command_line = [template.model_template_id,
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_train_data_roots(self, template):
        error_string = "ote train: error: the following arguments are required: --train-data-roots"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_val_ann_file(self, template):
        error_string = "ote train: error: the following arguments are required: --val-ann-files"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_val_data_roots(self, template):
        error_string = "ote train: error: the following arguments are required: --val-data-roots"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_no_save_model_to(self, template):
        error_string = "ote train: error: the following arguments are required: --save-model-to"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)
    
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_wrong_required_paths(self, template):
        error_string = "Path is not valid"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        for i in [4, 6, 8, 10, 12]:
            for case in wrong_paths.values():
                temp = deepcopy(command_line)
                temp[i] = case
                ret = ote_train_common(template, root, temp)
                assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_hpo_not_enabled(self, template):
        error_string = "Parameter --hpo-time-ratio must be used with --enable-hpo key"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        '--hpo-time-ratio', '4']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_wrong_hpo_value(self, template):
        error_string = "ote train: error: argument --hpo-time-ratio: invalid float value: 'STRING_VALUE'"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        '--enable-hpo',
                        '--hpo-time-ratio',
                        'STRING_VALUE']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train_wrong_hpo_value(self, template):
        error_string = "Parameter --hpo-time-ratio must not be negative"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}',
                        '--enable-hpo',
                        '--hpo-time-ratio',
                        '-1']
        ret = ote_train_common(template, root, command_line)
        assert error_string in str(ret.stderr)
