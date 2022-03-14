"""Tests for input parameters with OTE CLI train tool"""
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
import logging

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_cli.registry import Registry

from common import (
    create_venv,
    get_some_vars,
    args_paths,
    wrong_paths,
    ote_common
)

logger = logging.getLogger(__name__)

root = '/tmp/ote_cli/'
ote_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
external_path = os.path.join(ote_dir, "external")
# TODO check double quotes

params_values = []
params_ids = []
for back_end_ in ('DETECTION',
                  'CLASSIFICATION',
                  'ANOMALY_CLASSIFICATION',
                  'SEGMENTATION',
                  'ROTATED_DETECTION',
                  'INSTANCE_SEGMENTATION'):
    cur_templates = Registry(external_path).filter(task_type=back_end_).templates
    cur_templates_ids = [template.model_template_id for template in cur_templates]
    params_values += [(back_end_, t) for t in cur_templates]
    params_ids += [back_end_ + ',' + cur_id for cur_id in cur_templates_ids]


class TestOptimizeCommon:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values)
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_template(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: template"
        command_line = []
        ret = ote_common(template, root, 'optimize', command_line)
        logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
        logger.debug(f"Stdout: {ret['stdout']}\n")
        logger.debug(f"Stderr: {ret['stderr']}\n")
        logger.debug(f"Exit_code: {ret['exit_code']}\n")
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_train_ann_file(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --train-ann-files"
        command_line = [template.model_template_id,
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                        '--load-weights',
                        f'./exported_{template.model_template_id}/openvino.xml',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_common(template, root, 'optimize', command_line)
        logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
        logger.debug(f"Stdout: {ret['stdout']}\n")
        logger.debug(f"Stderr: {ret['stderr']}\n")
        logger.debug(f"Exit_code: {ret['exit_code']}\n")
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_train_data_roots(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --train-data-roots"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                        '--load-weights',
                        f'./exported_{template.model_template_id}/openvino.xml',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_common(template, root, 'optimize', command_line)
        logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
        logger.debug(f"Stdout: {ret['stdout']}\n")
        logger.debug(f"Stderr: {ret['stderr']}\n")
        logger.debug(f"Exit_code: {ret['exit_code']}\n")
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_val_ann_file(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --val-ann-files"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                        '--load-weights',
                        f'./exported_{template.model_template_id}/openvino.xml',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_common(template, root, 'optimize', command_line)
        logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
        logger.debug(f"Stdout: {ret['stdout']}\n")
        logger.debug(f"Stderr: {ret['stderr']}\n")
        logger.debug(f"Exit_code: {ret['exit_code']}\n")
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_val_data_roots(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --val-data-roots"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                        '--load-weights',
                        f'./exported_{template.model_template_id}/openvino.xml',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_common(template, root, 'optimize', command_line)
        logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
        logger.debug(f"Stdout: {ret['stdout']}\n")
        logger.debug(f"Stderr: {ret['stderr']}\n")
        logger.debug(f"Exit_code: {ret['exit_code']}\n")
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_save_model_to(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --save-model-to"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                        '--load-weights',
                        f'./exported_{template.model_template_id}/openvino.xml'
                        ]
        ret = ote_common(template, root, 'optimize', command_line)
        logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
        logger.debug(f"Stdout: {ret['stdout']}\n")
        logger.debug(f"Stderr: {ret['stderr']}\n")
        logger.debug(f"Exit_code: {ret['exit_code']}\n")
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_load_weights(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --load-weights"
        command_line = [template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        ret = ote_common(template, root, 'optimize', command_line)
        logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
        logger.debug(f"Stdout: {ret['stdout']}\n")
        logger.debug(f"Stderr: {ret['stderr']}\n")
        logger.debug(f"Exit_code: {ret['exit_code']}\n")
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_path_train_ann_file(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [template.model_template_id,
                            '--train-ann-file',
                            case,
                            '--train-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                            '--val-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                            '--val-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                            '--load-weights',
                            f'./exported_{template.model_template_id}/openvino.xml',
                            '--save-model-to',
                            f'./trained_{template.model_template_id}']
            ret = ote_common(template, root, 'optimize', command_line)
            logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
            logger.debug(f"Stdout: {ret['stdout']}\n")
            logger.debug(f"Stderr: {ret['stderr']}\n")
            logger.debug(f"Exit_code: {ret['exit_code']}\n")
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_paths_train_data_roots(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [template.model_template_id,
                            '--train-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                            '--train-data-roots',
                            case,
                            '--val-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                            '--val-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                            '--load-weights',
                            f'./exported_{template.model_template_id}/openvino.xml',
                            '--save-model-to',
                            f'./trained_{template.model_template_id}']
            ret = ote_common(template, root, 'optimize', command_line)
            logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
            logger.debug(f"Stdout: {ret['stdout']}\n")
            logger.debug(f"Stderr: {ret['stderr']}\n")
            logger.debug(f"Exit_code: {ret['exit_code']}\n")
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_paths_val_ann_file(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [template.model_template_id,
                            '--train-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                            '--train-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                            '--val-ann-file',
                            case,
                            '--val-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                            '--load-weights',
                            f'./exported_{template.model_template_id}/openvino.xml',
                            '--save-model-to',
                            f'./trained_{template.model_template_id}']
            ret = ote_common(template, root, 'optimize', command_line)
            logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
            logger.debug(f"Stdout: {ret['stdout']}\n")
            logger.debug(f"Stderr: {ret['stderr']}\n")
            logger.debug(f"Exit_code: {ret['exit_code']}\n")
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_paths_val_data_roots(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [template.model_template_id,
                            '--train-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                            '--train-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                            '--val-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                            '--val-data-roots',
                            case,
                            '--load-weights',
                            f'./exported_{template.model_template_id}/openvino.xml',
                            '--save-model-to',
                            f'./trained_{template.model_template_id}']
            ret = ote_common(template, root, 'optimize', command_line)
            logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
            logger.debug(f"Stdout: {ret['stdout']}\n")
            logger.debug(f"Stderr: {ret['stderr']}\n")
            logger.debug(f"Exit_code: {ret['exit_code']}\n")
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_saved_model_to(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [template.model_template_id,
                            '--train-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                            '--train-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                            '--val-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                            '--val-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                            '--load-weights',
                            f'./exported_{template.model_template_id}/openvino.xml',
                            '--save-model-to',
                            case]
            ret = ote_common(template, root, 'optimize', command_line)
            logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
            logger.debug(f"Stdout: {ret['stdout']}\n")
            logger.debug(f"Stderr: {ret['stderr']}\n")
            logger.debug(f"Exit_code: {ret['exit_code']}\n")
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_load_weights(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [template.model_template_id,
                            '--train-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                            '--train-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                            '--val-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                            '--val-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                            '--load-weights',
                            case,
                            '--save-model-to',
                            f'./trained_{template.model_template_id}']
            ret = ote_common(template, root, 'optimize', command_line)
            logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
            logger.debug(f"Stdout: {ret['stdout']}\n")
            logger.debug(f"Stderr: {ret['stderr']}\n")
            logger.debug(f"Exit_code: {ret['exit_code']}\n")
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_save_performance(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [template.model_template_id,
                            '--train-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--train-ann-file"])}',
                            '--train-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--train-data-roots"])}',
                            '--val-ann-file',
                            f'{os.path.join(ote_dir, args_paths["--val-ann-file"])}',
                            '--val-data-roots',
                            f'{os.path.join(ote_dir, args_paths["--val-data-roots"])}',
                            '--load-weights',
                            f'./exported_{template.model_template_id}/openvino.xml',
                            '--save-model-to',
                            f'./trained_{template.model_template_id}',
                            '--save-performance',
                            case]
            ret = ote_common(template, root, 'optimize', command_line)
            logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
            logger.debug(f"Stdout: {ret['stdout']}\n")
            logger.debug(f"Stderr: {ret['stderr']}\n")
            logger.debug(f"Exit_code: {ret['exit_code']}\n")
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
