"""
 Copyright (c) 2020-2021 Intel Corporation

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

from ote import REID_TOOLS


def collect_accuracy(path):
    """ Collects accuracy values in log file. """

    r1 = None
    r5 = None
    mAP = None
    r1_content = 'Rank-1 '
    r5_content = 'Rank-5 '
    map_content = 'mAP:'
    with open(path) as input_stream:
        for line in input_stream:
            candidate = line.strip()
            if r1_content in candidate:
                r1 = float(candidate.split(':')[-1].replace('%', ''))
            elif r5_content in candidate:
                r5 = float(candidate.split(':')[-1].replace('%', ''))
            elif map_content in candidate:
                mAP = float(candidate.split(':')[-1].replace('%', ''))

    return r1, r5, mAP


def mean_accuracy_eval(config_path, work_dir, snapshot, update_config, **kwargs):
    """ Computes mean accuracy. """
    def get_topk_dict(value, k=1):
        key_name = 'accuracy' if k == 1 else f'top_{k}_accuracy'
        return {
            'key': key_name, 'value': value, 'unit': '%', 'display_name': f'Top-{k} accuracy'
        }
    outputs = []

    if '--custom-roots' not in update_config:
        logging.warning('Passed empty path to data root folder. '
                        'Skipping accuracy calculation.')
        outputs.append(get_topk_dict(None, 1))
        outputs.append(get_topk_dict(None, 5))
    else:
        test_py_stdout = os.path.join(work_dir, 'test_py_stdout')

        subprocess.run(
            f'python3 {REID_TOOLS}/eval.py'
            f' --config-file {config_path}'
            f' {update_config}'
            f' test.evaluate True'
            f' mutual_learning.aux_configs []'
            f' | tee {test_py_stdout}',
            check=True, shell=True
        )

        r1, r5, _ = collect_accuracy(os.path.join(work_dir, 'test_py_stdout'))
        outputs.append(get_topk_dict(r1, 1))
        outputs.append(get_topk_dict(r5, 5))

    return outputs
