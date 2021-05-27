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

# pylint:disable=import-error
import os
import json
import sys
import csv
from shutil import make_archive
from typing import Tuple, List

import pytest
import subprocess
import re
import shlex
from collections import OrderedDict
from pathlib import Path
from yattag import Doc

BG_COLOR_GREEN_HEX = 'ccffcc'
BG_COLOR_RED_HEX = 'ffcccc'
param_list = []
ids_list = []
DIFF_TARGET_MIN_GLOBAL = -0.1
DIFF_TARGET_MAX_GLOBAL = 0.1

TEST_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = TEST_ROOT.parent.parent
env = os.environ.copy()
if 'INTEL_OPENVINO_DIR' in env:
    OPENVINO_DIR = Path(env['INTEL_OPENVINO_DIR'])
else:
    OPENVINO_DIR = PROJECT_ROOT.parent / 'intel' / 'openvino'
    if not OPENVINO_DIR.is_dir():
        OPENVINO_DIR = PROJECT_ROOT.parent / 'intel' / 'openvino_2021'
        if not OPENVINO_DIR.is_dir():
            raise Exception("OpenVino path not found!")
ACC_CHECK_DIR = OPENVINO_DIR / 'deployment_tools' / 'open_model_zoo' / 'tools' / 'accuracy_checker'
MO_DIR = OPENVINO_DIR / 'deployment_tools' / 'model_optimizer'
ACC_CHECK_VENV_DIR = PROJECT_ROOT / 'acc_check'
acc_check_activate_string = f'source {ACC_CHECK_VENV_DIR}/bin/activate && source {OPENVINO_DIR}/bin/setupvars.sh'

param_list = []
ids_list = []
row_dict = OrderedDict()
color_dict = OrderedDict()


@pytest.fixture(autouse=True, scope="module")
def setup_ac():
    subprocess.run(f'virtualenv -ppython3.7 {ACC_CHECK_VENV_DIR}', cwd=PROJECT_ROOT, check=True, shell=True)

    #WAD to avoid Accuracy Checker install error
    subprocess.run(f'{acc_check_activate_string} && {ACC_CHECK_VENV_DIR}/bin/pip install scikit-image!=0.18.2rc1',
                   cwd=ACC_CHECK_DIR, check=True, shell=True, executable='/bin/bash')

    subprocess.run(f'{acc_check_activate_string} && {ACC_CHECK_VENV_DIR}/bin/pip install pycocotools',
                   cwd=ACC_CHECK_DIR, check=True, shell=True, executable='/bin/bash')
    subprocess.run(f'{acc_check_activate_string} && {ACC_CHECK_VENV_DIR}/bin/python setup.py install',
                   cwd=ACC_CHECK_DIR, check=True, shell=True, executable='/bin/bash')



def run_cmd(comm: str, cwd: str, venv=None) -> Tuple[int, str]:
    print()
    print(comm)
    print()
    com_line = shlex.split(comm)
    print(com_line)
    cmd_env = os.environ.copy()
    if venv:
        cmd_env['VIRTUAL_ENV'] = str(venv)
        cmd_env['PATH'] = str(f'{venv}/bin') + ':' + cmd_env['PATH']
    result = subprocess.Popen(com_line, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd=cwd, env=cmd_env)
    exit_code = result.poll()

    def process_line(decoded_line: str, error_lines: List):
        if re.search('Error|(No module named)', decoded_line):
            # WA for tensorboardX multiprocessing bug (https://github.com/lanpa/tensorboardX/issues/598)
            if not re.search('EOFError', decoded_line):
                error_lines.append(decoded_line)
        if decoded_line != '':
            print(decoded_line)

    error_lines = []
    while exit_code is None:
        decoded_line = result.stdout.readline().decode('utf-8').strip()
        process_line(decoded_line, error_lines)
        exit_code = result.poll()

    # The process may exit before the first process_line is executed, handling this case here
    outs, _ = result.communicate()
    remaining_lines = outs.decode('utf-8').strip().split('\n')
    for output_line in remaining_lines:
        process_line(output_line, error_lines)

    err_string = "\n".join(error_lines) if error_lines else None
    return exit_code, err_string


def read_metric(metric_file):
    metric = 0
    if os.path.exists(metric_file):
        input_csv_file = csv.DictReader(open(metric_file))
        for row in input_csv_file:
            metric = round(float(row['metric_value']) * 100, 2)
    return metric


def make_table_row(expected_, key, error_message, metric, diff_target):
    if metric is not None:
        row = [str(key), str(expected_), str(metric), str(diff_target), str('-')]
    else:
        row = [str(key), str(expected_), str('Not executed'), str('-'), str(error_message)]
    return row


def get_export_test_params(model, problem_name, test_id, domain_name, is_alt_ssd_export=None):
    workdir = PROJECT_ROOT / model
    sub_folder = problem_name.replace('-', '_')
    test_folder = str(f'output_export_tests_{sub_folder}.{test_id}')
    config_dir = workdir / domain_name / problem_name / model
    ir_dir = config_dir / test_folder / 'gpu_export'
    if is_alt_ssd_export:
        workdir = PROJECT_ROOT / model.replace('-alt-ssd-export', '')
        sub_folder = problem_name.replace('-', '_')
        if os.path.isdir(workdir):
            test_id = str(test_id).replace('_alt_ssd', '')
        test_folder = str(f'output_export_tests_{sub_folder}.{test_id}')
        config_dir = workdir / domain_name / problem_name / model.replace('-alt-ssd-export', '')
        ir_dir = config_dir / test_folder / 'gpu_export' / 'alt_ssd_export'
    return workdir, config_dir, ir_dir, test_id


def write_results_table(init_table_string):
    doc, tag, text = Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    with tag('p'):
        text('legend: ')
    with tag('p'):
        with tag('span', style='Background-color: #{}'.format(BG_COLOR_GREEN_HEX)):
            text('Thresholds Expected are passed')
    with tag('p'):
        with tag('span', style='Background-color: #{}'.format(BG_COLOR_RED_HEX)):
            text('Thresholds for Expected are failed, or model was not evaluate')
    with tag('p'):
        text('If Reference FP32 value in parentheses, it takes from "target" field of .json file')
    with tag('table', border='1', cellpadding='5', style='border-collapse: collapse; border: 1px solid;'):
        with tag('tr'):
            for i in init_table_string:
                with tag('td'):
                    text(i)
        for key in row_dict:
            with tag('tr', bgcolor='{}'.format(color_dict[key])):
                for i in row_dict[key]:
                    if i is None:
                        i = '-'
                    with tag('td'):
                        text(i)
    f = open(PROJECT_ROOT / 'results.html', 'w')
    f.write(doc.getvalue())
    f.close()


def get_test_params():
    global param_list
    global ids_list
    if param_list or ids_list:
        return param_list, ids_list
    eval_config = json.load(open(TEST_ROOT / 'ote_accuracy_validation.json'), object_pairs_hook=OrderedDict)
    for domain_name in eval_config:
        model_type = eval_config[domain_name]
        for problem_name in model_type:
            model_dict = model_type[problem_name]
            for model in model_dict:
                test_id = model_dict[model].get('test_id', {})
                expected = model_dict[model].get('target', {})
                alt_export = model_dict[model].get('subfolder')
                param_list.append([model, test_id, domain_name, problem_name, expected, alt_export])
                ids_list.append(model)
    return param_list, ids_list


@pytest.mark.parametrize('model_, test_id_, domain_name_, problem_name_, expected_, alt_export_', get_test_params()[0],
                         ids=get_test_params()[1])
def test_eval(data_dir, model_, test_id_, domain_name_, problem_name_, expected_, alt_export_):
    os.environ['INTEL_OPENVINO_DIR'] = str(OPENVINO_DIR)
    config_name = 'accuracy-check'
    metric_value = None
    diff_target = None
    err_str = None
    exit_code = 0
    workdir, config_dir, ir_dir, test_id_ = get_export_test_params(model_,
                                                                   problem_name_,
                                                                   test_id_,
                                                                   domain_name_,
                                                                   alt_export_)
    if not os.path.isdir(ir_dir):
        ote_cmd_string = f'{sys.executable} tests/run_model_templates_tests.py' \
                         f' --verbose' \
                         f'  --topic export' \
                         f'  --test-id-filter' \
                         f' \"*_on_gpu*\"' \
                         f' --test-id-filter \"*{test_id_}*\" ' \
                         f' --workdir {workdir}'
        exit_code, err_str = run_cmd(ote_cmd_string, PROJECT_ROOT)
    if exit_code == 0:
        if alt_export_:
            config_name += '_alt-ssd-export'
            if not os.path.isfile(f"{model_.replace('-alt-ssd-export', '')}.zip"):
                make_archive(model_, 'zip', ir_dir)
        else:
            make_archive(model_, 'zip', ir_dir)
        ac_cmd_string = f'accuracy_check' \
                        f' -c {config_dir}/{config_name}.yml' \
                        f' -s {data_dir}' \
                        f' -d dataset_definitions.yml' \
                        f' -td CPU' \
                        f' -m {ir_dir}' \
                        f' --csv_result {PROJECT_ROOT}/{model_}.csv'
        exit_code, err_str = run_cmd(ac_cmd_string, ACC_CHECK_DIR, ACC_CHECK_VENV_DIR)
        if exit_code == 0:
            metric_value = read_metric(f'{PROJECT_ROOT}/{model_}.csv')
            diff_target = round((metric_value - expected_), 2)
    row_dict[model_] = make_table_row(expected_,
                                      model_,
                                      err_str,
                                      metric_value,
                                      diff_target)
    if diff_target is None or diff_target > DIFF_TARGET_MAX_GLOBAL or diff_target < DIFF_TARGET_MIN_GLOBAL:
        color_dict[model_] = BG_COLOR_RED_HEX
        pytest.fail(err_str)
    else:
        color_dict[model_] = BG_COLOR_GREEN_HEX


@pytest.fixture(autouse=True, scope='module')
def results(data_dir):
    yield
    if data_dir:
        header = ['Model', 'Expected', 'Measured', 'Diff Expected', 'Error']
        write_results_table(header)
