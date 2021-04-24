"""
 Copyright (c) 2020 Intel Corporation

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

from ote import MMACTION_TOOLS


def collect_accuracy(path):
    """ Collects accuracy values in log file. """

    accuracies = []
    content = '/mean_top1_acc:'
    with open(path) as input_stream:
        for line in input_stream:
            candidate = line.strip()
            if content in candidate:
                accuracies.append(float(candidate.split(' ')[-1]))

    return accuracies


def mean_accuracy_eval(config_path, work_dir, snapshot, update_config, **kwargs):
    """ Computes mean accuracy. """

    outputs = []

    if not(update_config['data.test.ann_file'] and update_config['root_dir']):
        logging.warning('Passed empty path to annotation file or data root folder. '
                        'Skipping accuracy calculation.')
        outputs.append({
            'key': 'accuracy', 'value': None, 'unit': '%', 'display_name': 'Top-1 accuracy'
        })
    else:
        test_py_stdout = os.path.join(work_dir, 'test_py_stdout')

        update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
        update_config = f' --update_config {update_config}' if update_config else ''
        update_config = update_config.replace('"', '\\"')

        if snapshot.split('.')[-1] in {'xml', 'bin', 'onnx'}:
            if snapshot.split('.')[-1] == 'bin':
                snapshot = '.'.join(snapshot.split('.')[:-1]) + '.xml'
            tool = 'test_exported.py'
        else:
            tool = 'test.py'

        subprocess.run(
            f'python3 {MMACTION_TOOLS}/{tool}'
            f' {config_path} {snapshot}'
            f' --eval mean_top_k_accuracy'
            f'{update_config}'
            f' | tee {test_py_stdout}',
            check=True, shell=True
        )

        average_precision = collect_accuracy(os.path.join(work_dir, 'test_py_stdout'))[0]
        outputs.append({
            'key': 'accuracy', 'value': 1e2 * average_precision, 'unit': '%', 'display_name': 'Top-1 accuracy'
        })

    return outputs
