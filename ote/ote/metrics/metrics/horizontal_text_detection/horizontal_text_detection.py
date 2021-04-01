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

import os
import subprocess

from ote import MMDETECTION_TOOLS
from ote.metrics.detection.common import collect_ap


def collect_f1(path):
    """ Collects precision, recall and f1 score values in log file. """
    metrics = ['recall', 'precision', 'hmean']
    content = []
    result = []
    with open(path) as read_file:
        content += [line.split() for line in read_file]
        for line in content:
            if (len(line) > 2) and (line[0] == 'Text'):
                for word in line[2:]:
                    for metric in metrics:
                        if word.startswith(metric):
                            result.append(float(word.replace(metric + '=', '')))
    return result


def coco_eval(config_path, work_dir, snapshot, update_config, show_dir, **kwargs):
    """ Computes metrics: precision, recall, hmean and COCO AP. """

    outputs = []

    res_pkl = os.path.join(work_dir, 'res.pkl')
    test_py_stdout = os.path.join(work_dir, 'test_py_stdout')
    update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
    update_config = f' --update_config {update_config}' if update_config else ''
    show_dir = f' --show-dir {show_dir}' if show_dir else ''
    if snapshot.split('.')[-1] in {'xml', 'bin', 'onnx'}:
        if snapshot.split('.')[-1] == 'bin':
            snapshot = '.'.join(snapshot.split('.')[:-1]) + '.xml'
        tool = 'test_exported.py'
    else:
        tool = 'test.py'

    subprocess.run(
        f'python3 {MMDETECTION_TOOLS}/{tool}'
        f' {config_path} {snapshot}'
        f' --out {res_pkl} --eval f1 bbox'
        f'{show_dir}{update_config}'
        f' | tee {test_py_stdout}',
        check=True, shell=True
    )

    hmean = collect_f1(os.path.join(work_dir, 'test_py_stdout'))
    outputs.append({
        'key': 'f1', 'value': hmean[2] * 100, 'unit': '%', 'display_name': 'F1-score'
    })
    outputs.append({
        'key': 'recall', 'value': hmean[0] * 100, 'unit': '%', 'display_name': 'Recall'
    })
    outputs.append({
        'key': 'precision', 'value': hmean[1] * 100, 'unit': '%', 'display_name': 'Precision'
    })

    average_precision = collect_ap(os.path.join(work_dir, 'test_py_stdout'))[0]
    outputs.append({
        'key': 'bbox', 'value': average_precision * 100, 'unit': '%', 'display_name': 'AP @ [IoU=0.50:0.95]'
    })

    return outputs
