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
from shutil import copy, make_archive, unpack_archive
from typing import Tuple, List

import pytest
import subprocess
import re
import shlex
from collections import OrderedDict
from pathlib import Path

BG_COLOR_GREEN_HEX = 'ccffcc'
BG_COLOR_RED_HEX = 'ffcccc'
param_list = []
ids_list = []
DIFF_TARGET_MIN_GLOBAL = -0.1
DIFF_TARGET_MAX_GLOBAL = 0.1

TEST_ROOT = Path(__file__).parent
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
ACC_CHECK_VENV_DIR = PROJECT_ROOT / 'acc_check'
acc_check_activate_string = f'source {ACC_CHECK_VENV_DIR}/bin/activate && source {OPENVINO_DIR}/bin/setupvars.sh'


param_list = []
ids_list = []
row_dict = OrderedDict()
color_dict = OrderedDict()


@pytest.fixture(scope="module")
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
    com_line = shlex.split(comm)
    print(com_line, '\n')
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


def create_pot_config(model_name, ac_config):
    pot_config = json.load(
        open(TEST_ROOT / 'pot_config.json'), object_pairs_hook=OrderedDict)
    pot_config["model"]["model_name"] = model_name
    pot_config["model"]["model"] = str((PROJECT_ROOT / 'ir_model').absolute()) + '/model.xml'
    pot_config["model"]["weights"] = str((PROJECT_ROOT / 'ir_model').absolute()) + '/model.bin'
    pot_config["engine"]["type"] = 'accuracy_checker'
    pot_config["engine"]["config"] = str((PROJECT_ROOT / 'ir_model' / Path(ac_config)).absolute())
    with open(f'{model_name}.json', 'w') as outfile:
        json.dump(pot_config, outfile, indent=4)


def write_error_in_csv(error_message, filename, model_name):
    error_message = 'Error ' + error_message[:80].replace("\n", '')
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['model', 'launcher', 'device', 'dataset', 'tags', 'metric_name', 'metric_type',
                        'metric_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'model': model_name, 'launcher': '-', 'device': '-', 'dataset': '-', 'tags': '-',
                            'metric_name': '-', 'metric_type': '-', 'metric_value': error_message})


def get_test_params():
    global param_list
    global ids_list
    if param_list or ids_list:
        return param_list, ids_list
    eval_config = json.load(
        open(TEST_ROOT / 'ote_accuracy_validation.json'), object_pairs_hook=OrderedDict)
    for domain_name in eval_config:
        model_type = eval_config[domain_name]
        for problem_name in model_type:
            model_dict = model_type[problem_name]
            for model in model_dict:
                test_id = model_dict[model].get('test_id', {})
                alt_export = model_dict[model].get('subfolder')
                param_list.append(
                    [model, test_id, domain_name, problem_name, alt_export])
                ids_list.append(model)
    return param_list, ids_list


@pytest.mark.parametrize('model_, test_id_, domain_name_, problem_name_, alt_export_', get_test_params()[0],
                         ids=get_test_params()[1])
def test_export(data_dir, model_, test_id_, domain_name_, problem_name_, alt_export_):
    os.environ['INTEL_OPENVINO_DIR'] = str(OPENVINO_DIR)
    exit_code = 0
    err_str = None
    workdir = PROJECT_ROOT / model_
    sub_folder = problem_name_.replace('-', '_')
    test_folder = str(f'output_export_tests_{sub_folder}.{test_id_}')
    config_dir = workdir / domain_name_ / problem_name_ / model_
    ir_dir = config_dir / test_folder / 'gpu_export'
    if not alt_export_:
        ote_cmd_string = f'{sys.executable} tests/run_model_templates_tests.py' \
                         f' --verbose' \
                         f'  --topic export' \
                         f'  --test-id-filter' \
                         f' \"*_on_gpu*\"' \
                         f' --test-id-filter \"*{test_id_}*\" ' \
                         f' --workdir {workdir}'
        exit_code, err_str = run_cmd(ote_cmd_string, PROJECT_ROOT)
    else:
        pytest.skip()
    # Copy AC configs to IR folder
    src_files = os.listdir(config_dir)
    for file_name in src_files:
        if re.search('accuracy-check', file_name):
            copy((config_dir / file_name), ir_dir)
    # Archive IR folder
    if ir_dir.is_dir():
        make_archive(model_, 'zip', ir_dir)
    if exit_code != 0:
        pytest.fail(err_str)
    assert Path(f'{model_}.zip').is_file()


# pylint:disable=redefined-outer-name
@pytest.mark.parametrize('model_, test_id_, domain_name_, problem_name_, alt_export_', get_test_params()[0],
                         ids=get_test_params()[1])
def test_eval(setup_ac, data_dir, model_, test_id_, domain_name_, problem_name_, alt_export_):
    os.environ['INTEL_OPENVINO_DIR'] = str(OPENVINO_DIR)
    base_model = model_
    config_name = 'accuracy-check.yml'
    model_folder = config_folder = PROJECT_ROOT / 'ir_model'
    if alt_export_:
        base_model = model_.replace('-alt-ssd-export', '')
        config_name = 'accuracy-check_alt-ssd-export.yml'
    model_archive = f'{base_model}.zip'
    if (PROJECT_ROOT / model_archive).is_file():
        unpack_archive(model_archive, 'ir_model', 'zip')
    create_pot_config(model_, config_name)

    ac_cmd_string = 'accuracy_check' \
                    f' -c {config_folder}/{config_name}' \
                    f' -s {data_dir}' \
                    ' --progress print' \
                    f' -d dataset_definitions.yml' \
                    f' -td CPU' \
                    f' -m {model_folder}' \
                    f' --csv_result {PROJECT_ROOT}/report.csv'
    exit_code, err_str = run_cmd(ac_cmd_string, ACC_CHECK_DIR, ACC_CHECK_VENV_DIR)
    if exit_code != 0:
        write_error_in_csv(err_str, 'report', model_)
        pytest.fail(err_str)
    assert (PROJECT_ROOT / 'report.csv').is_file()
