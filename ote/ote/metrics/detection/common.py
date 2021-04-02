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

from ote import MMDETECTION_TOOLS


def collect_ap(path):
    """ Collects average precision values in log file. """

    average_precisions = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file]
        for line in content:
            if line.startswith(beginning):
                average_precisions.append(float(line.replace(beginning, '')))
    return average_precisions


def update_outputs(outputs, metric_keys, metric_names, metric_values):
    assert len(metric_values) == len(metric_names) == len(metric_keys), \
        f'{metric_values} vs {metric_names} vs {metric_keys}'
    for key, name, value in zip(metric_keys, metric_names, metric_values):
        assert 0 <= value <= 1.0, f'{key} = {value}'
        outputs.append(
            {'key': key, 'value': value * 100, 'unit': '%', 'display_name': name})


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
        f'python3 {MMDETECTION_TOOLS}/{tool}'
        f' {config_path} {snapshot}'
        f' --out {res_pkl} --eval {metrics}'
        f'{show_dir}{update_config}'
        f' | tee {test_py_stdout}',
        check=True, shell=True
    )

    return test_py_stdout


def coco_ap_eval(config_path, work_dir, snapshot, update_config, show_dir='',
                 metric_names=('AP @ [IoU=0.50:0.95]', ), metrics='bbox', **kwargs):
    """ Computes COCO AP. """

    metric_keys = metrics.split(' ')
    assert len(metric_keys) == len(metric_names), f'{len(metric_keys)} != {len(metric_names)}'
    allowed_metric_keys = {'bbox', 'segm'}
    assert all(x in allowed_metric_keys for x in metric_keys)
    outputs = []
    if not(update_config['data.test.ann_file'] and update_config['data.test.img_prefix']):
        logging.warning('Passed empty path to annotation file or data root folder. '
                        'Skipping AP calculation.')
        update_outputs(outputs, metric_keys, metric_names, [None for _ in metric_keys])
    else:
        test_py_stdout = run_test_script(config_path, work_dir, snapshot,
                                         update_config, show_dir, metrics)

        average_precision = collect_ap(test_py_stdout)
        update_outputs(outputs, metric_keys, metric_names, average_precision)

    return outputs


def coco_ap_eval_det(config_path, work_dir, snapshot, update_config, show_dir='', **kwargs):
    return coco_ap_eval(config_path, work_dir, snapshot, update_config, show_dir)


def coco_ap_eval_segm(config_path, work_dir, snapshot, update_config, show_dir='', **kwargs):
    return coco_ap_eval(
        config_path, work_dir, snapshot, update_config, show_dir,
        metric_names=['Bbox AP @ [IoU=0.50:0.95]', 'Segm AP @ [IoU=0.50:0.95]'],
        metrics='bbox segm')
