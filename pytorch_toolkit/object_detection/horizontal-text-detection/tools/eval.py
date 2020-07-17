# Copyright (C) 2020 Intel Corporation
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

# pylint: disable=C0301,W0622,R0914,R0913

import argparse
import os
import subprocess
import sys
sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}/../../')

from tools.misc import evaluate, collect_ap

MMDETECTION_TOOLS = f'{os.path.dirname(__file__)}/../../../../external/mmdetection/tools'


def parse_args():
    """ Parses input args. """

    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='A path to model training configuration file (.py).')
    parser.add_argument('snapshot',
                        help='A path to pre-trained snapshot (.pth).')
    parser.add_argument('out',
                        help='A path to output file where models metrics will be saved (.yml).')
    parser.add_argument('--update_config',
                        help='Update configuration file by parameters specified here.'
                             'Use quotes if you are going to change several params.',
                        default='')

    return parser.parse_args()


def collect_f1(path):
    """ Collects precision, recall and f1 score values in log file. """
    metrics = ['recall', 'precision', 'hmean']
    content = []
    result = []
    with open(path) as read_file:
        content += [line.split() for line in read_file.readlines()]
        for line in content:
            if (len(line) > 2) and (line[0] == 'Text'):
                for word in line[2:]:
                    for metric in metrics:
                        if word.startswith(metric):
                            result.append(float(word.replace(metric + '=', '')))
    return result


def coco_eval(config_path, work_dir, snapshot, outputs, update_config):
    """ Computes metrics: precision, recall, hmean and COCO AP. """

    res_pkl = os.path.join(work_dir, 'res.pkl')
    with open(os.path.join(work_dir, 'test_py_stdout'), 'w') as test_py_stdout:
        update_config = f' --update_config {update_config}' if update_config else ''
        subprocess.run(
            f'python {MMDETECTION_TOOLS}/test.py'
            f' {config_path} {snapshot}'
            f' --out {res_pkl} --eval f1 bbox{update_config}'.split(' '), stdout=test_py_stdout,
            check=True)
    hmean = collect_f1(os.path.join(work_dir, 'test_py_stdout'))
    print(hmean)
    print(os.path.join(work_dir, 'test_py_stdout'))
    outputs.append({'key': 'f1', 'value': hmean[2] * 100, 'unit': '%', 'display_name': 'F1-score'})
    outputs.append(
        {'key': 'recall', 'value': hmean[0] * 100, 'unit': '%', 'display_name': 'Recall'})
    outputs.append(
        {'key': 'precision', 'value': hmean[1] * 100, 'unit': '%', 'display_name': 'Precision'})

    average_precision = collect_ap(os.path.join(work_dir, 'test_py_stdout'))[0]
    outputs.append({'key': 'ap', 'value': average_precision * 100, 'unit': '%',
                    'display_name': 'AP @ [IoU=0.50:0.95]'})
    return outputs


def main(config, snapshot, out, update_config):
    """ Main function. """

    metrics_functions = [coco_eval]
    evaluate(config, snapshot, out, update_config, metrics_functions)


if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.snapshot, args.out, args.update_config)
