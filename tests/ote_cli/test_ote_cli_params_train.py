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

from copy import deepcopy
import os
from subprocess import run
import shutil

import pytest

from ote_cli.registry import Registry
from common import args, wrong_paths

ote_dir = os.getcwd()


@pytest.fixture()
def templates(algo_be):
    return Registry('external').filter(task_type=algo_be).templates


def test_ote_train_no_train_ann_file(templates):
    error_string = "ote train: error: the following arguments are required: --train-ann-files"
    for template in templates:
        command_line = ['ote',
                        'train',
                        template.model_template_id,
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        assert error_string in str(run(command_line, capture_output=True).stderr)


def test_ote_train_no_train_data_roots(templates):
    error_string = "ote train: error: the following arguments are required: --train-data-roots"
    for template in templates:
        command_line = ['ote',
                        'train',
                        template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        assert error_string in str(run(command_line, capture_output=True).stderr)


def test_ote_train_no_val_ann_file(templates):
    error_string = "ote train: error: the following arguments are required: --val-ann-files"
    for template in templates:
        command_line = ['ote',
                        'train',
                        template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        assert error_string in str(run(command_line, capture_output=True).stderr)


def test_ote_train_no_val_data_roots(templates):
    error_string = "ote train: error: the following arguments are required: --val-data-roots"
    for template in templates:
        command_line = ['ote',
                        'train',
                        template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--save-model-to',
                        f'./trained_{template.model_template_id}']
        assert error_string in str(run(command_line, capture_output=True).stderr)


def test_ote_train_no_save_model_to(templates):
    error_string = "ote train: error: the following arguments are required: --save-model-to"
    for template in templates:
        command_line = ['ote',
                        'train',
                        template.model_template_id,
                        '--train-ann-file',
                        f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                        '--train-data-roots',
                        f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                        '--val-ann-file',
                        f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                        '--val-data-roots',
                        f'{os.path.join(ote_dir, args["--val-data-roots"])}']
        assert error_string in str(run(command_line, capture_output=True).stderr)


def test_ote_train_wrong_required_paths(templates):
    for template in templates:
        command_line = ['ote',
                        'train',
                        template.model_template_id,
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
                assert "Path is not valid" in str(run(temp, capture_output=True).stderr)


def test_ote_train_hpo_not_enabled(templates):
    expected_error = "Parameter --hpo-time-ratio must be used with --enable-hpo key"
    for template in templates:
        command_line = ['ote',
                        'train',
                        template.model_template_id,
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
        assert expected_error in str(run(command_line, capture_output=True).stderr)


def test_ote_train_wrong_hpo_value(templates):
    for template in templates:
        command_line = ['ote',
                        'train',
                        template.model_template_id,
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
        expected_error_line = "ote train: error: argument --hpo-time-ratio: invalid float value: 'STRING_VALUE'"
        assert expected_error_line in str(run(command_line, capture_output=True).stderr)
        temp = deepcopy(command_line)
        temp[-1] = "-1"
        expected_error_line = "Parameter --hpo-time-ratio must not be negative"
        assert expected_error_line in str(run(temp, capture_output=True).stderr)


def test_ote_train_no_template():
    error_string = "ote train: error: the following arguments are required: template"
    command_line = ['ote',
                    'train']
    assert error_string in str(run(command_line, capture_output=True).stderr)
