"""
 Copyright (c) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
import os
import subprocess

from ote import MMPOSE_TOOLS


def collect_NME(path):
    """ Collects NME values in log file. """

    NME = []
    beginning = "OrderedDict([('NME', "
    with open(path) as read_file:
        content = [line.strip() for line in read_file]
        for line in content:
            if line.startswith(beginning):
                NME.append(float(line.replace(beginning, '')[:-3]))
    return NME


def update_outputs(outputs, metric_keys, metric_names, metric_values):
    assert len(metric_values) == len(metric_names) == len(metric_keys), \
        f'{metric_values} vs {metric_names} vs {metric_keys}'
    for key, name, value in zip(metric_keys, metric_names, metric_values):
        outputs.append(
            {'key': key, 'value': value, 'unit': 'value', 'display_name': name})


def run_test_script(config_path, work_dir, snapshot, update_config, show_dir, metrics):
    res_pkl = os.path.join(work_dir, 'res.pkl')
    test_py_stdout = os.path.join(work_dir, 'test_py_stdout')

    update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
    update_config = f' --update_config {update_config}' if update_config else ''
    update_config = update_config.replace('"', '\\"')
    show_dir = f' --show-dir {show_dir}' if show_dir else ''

    if snapshot.split('.')[-1] in {'xml', 'bin', 'onnx'}:
        if snapshot.split('.')[-1] == 'bin':
            snapshot = '.'.join(snapshot.split('.')[:-1]) + '.xml'
        tool = 'test_exported.py'
    else:
        tool = 'test.py'

    subprocess.run(
        f'python3 {MMPOSE_TOOLS}/{tool}'
        f' {config_path} {snapshot}'
        f' --out {res_pkl} --eval {metrics}'
        f'{show_dir}{update_config}'
        f' | tee {test_py_stdout}',
        check=True, shell=True
    )

    return test_py_stdout


def coco_nme_eval(config_path, work_dir, snapshot, update_config, show_dir='',
                 metric_names=['NME', ], metrics='NME', **kwargs):
    """ Computes COCO NME. """

    metric_keys = metrics.split(' ')
    assert len(metric_keys) == len(metric_names), f'{len(metric_keys)} != {len(metric_names)}'
    allowed_metric_keys = {'NME', }
    assert all([x in allowed_metric_keys for x in metric_keys])
    outputs = []
    test_py_stdout = run_test_script(config_path, work_dir, snapshot,
                                     update_config, show_dir, metrics)
    nme = collect_NME(test_py_stdout)
    update_outputs(outputs, metric_keys, metric_names, nme)

    return outputs
