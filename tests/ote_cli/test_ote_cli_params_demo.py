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

from ote_cli.registry import Registry


from common import wrong_paths


ote_dir = os.getcwd()


@pytest.fixture()
def templates(algo_be):
    return Registry('external').filter(task_type=algo_be).templates


def test_ote_demo_no_weights(templates):
    expected_error = "ote demo: error: the following arguments are required: --load-weights"
    for template in templates:
        command_line = ['ote',
                        'demo',
                        template.model_template_id,
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}']
        assert expected_error in str(run(command_line, capture_output=True).stderr)


def test_ote_demo_no_input(templates):
    expected_error = "ote demo: error: the following arguments are required: -i/--input"
    for template in templates:
        command_line = ['ote',
                        'demo',
                        template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth']
        assert expected_error in str(run(command_line, capture_output=True).stderr)


def test_ote_demo_wrong_weights(templates):
    for template in templates:
        command_line = ['ote',
                        'demo',
                        template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}']
        for case in wrong_paths.values():
            temp = deepcopy(command_line)
            temp[4] = case
            assert "Path is not valid" in str(run(temp, capture_output=True).stderr)


def test_ote_demo_wrong_input(templates):
    expected_error_line = "ote demo: error: argument -i/--input: expected one argument"
    for template in templates:
        command_line = ['ote',
                        'demo',
                        template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--input']
        assert expected_error_line in str(run(command_line, capture_output=True).stderr)


def test_ote_demo_fit_size_wrong_input(templates):
    for template in templates:
        command_line = ['ote',
                        'demo',
                        template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--fit-to-size']
        expected_error_line = "ote demo: error: argument --fit-to-size: expected 2 arguments"
        assert expected_error_line in str(run(command_line, capture_output=True).stderr)
        temp = deepcopy(command_line)
        temp += ['0.0', '0.0']
        expected_error_line = "ote demo: error: argument --fit-to-size: invalid int value: '0.0'"
        assert expected_error_line in str(run(temp, capture_output=True).stderr)
        temp[-1], temp[-2] = "-1", "1"
        expected_error_line = "Both values of --fit_to_size parameter must be > 0"
        assert expected_error_line in str(run(temp, capture_output=True).stderr)


def test_ote_demo_delay_wrong_input(templates):

    for template in templates:
        command_line = ['ote',
                        'demo',
                        template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--delay',
                        '0.0']
        expected_error_line = "ote demo: error: argument --delay: invalid int value: '0.0'"
        assert expected_error_line in str(run(command_line, capture_output=True).stderr)
        temp = deepcopy(command_line)
        temp[-1] = "-1"
        expected_error_line = "Value of --delay parameter must not be negative"
        assert expected_error_line in str(run(temp, capture_output=True).stderr)


def test_ote_demo_no_template():
    error_string = "ote demo: error: the following arguments are required: template"
    command_line = ['ote',
                    'demo']
    assert error_string in str(run(command_line, capture_output=True).stderr)
