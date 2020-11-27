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
from ote.metrics.detection.common import collect_ap


def update_outputs(outputs, ap_values):
    display_names = ['Bbox AP @ [IoU=0.50:0.95]', 'Segm AP @ [IoU=0.50:0.95]']
    assert len(ap_values) == len(display_names)
    for name, ap in zip(display_names, ap_values):
        outputs.append(
            {'key': 'ap', 'value': ap, 'unit': '%', 'display_name': name})


def coco_ap_eval(config_path, work_dir, snapshot, update_config, show_dir='', **kwargs):
    """ Computes COCO AP. """

    outputs = []

    if not(update_config['data.test.ann_file'] and update_config['data.test.img_prefix']):
        logging.warning('Passed empty path to annotation file or data root folder. '
                        'Skipping AP calculation.')
        update_outputs(outputs, [None, None])
    else:
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
            f'python {MMDETECTION_TOOLS}/{tool}'
            f' {config_path} {snapshot}'
            f' --out {res_pkl} --eval bbox segm'
            f'{show_dir}{update_config}'
            f' | tee {test_py_stdout}',
            check=True, shell=True
        )

        average_precision = collect_ap(os.path.join(work_dir, 'test_py_stdout'))
        update_outputs(outputs, average_precision)

    return outputs
