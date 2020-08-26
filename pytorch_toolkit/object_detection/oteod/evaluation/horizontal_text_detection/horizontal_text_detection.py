# pylint: disable=R0913

import logging
import os
import subprocess

from oteod import MMDETECTION_TOOLS
from oteod.evaluation.common import collect_ap, evaluate_internal


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


def coco_eval(config_path, work_dir, snapshot, outputs, update_config, show_dir):
    """ Computes metrics: precision, recall, hmean and COCO AP. """

    res_pkl = os.path.join(work_dir, 'res.pkl')
    with open(os.path.join(work_dir, 'test_py_stdout'), 'w') as test_py_stdout:
        update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
        update_config = f' --update_config {update_config}' if update_config else ''
        show_dir = f' --show-dir {show_dir}' if show_dir else ''
        subprocess.run(
            f'python {MMDETECTION_TOOLS}/test.py'
            f' {config_path} {snapshot}'
            f' --out {res_pkl} --eval f1 bbox'
            f'{show_dir}{update_config}'.split(' '), stdout=test_py_stdout,
            check=True)
    hmean = collect_f1(os.path.join(work_dir, 'test_py_stdout'))
    with open(os.path.join(work_dir, 'test_py_stdout')) as test_py_stdout:
        logging.info(''.join(test_py_stdout.readlines()))
    outputs.append({'key': 'f1', 'value': hmean[2] * 100, 'unit': '%', 'display_name': 'F1-score'})
    outputs.append(
        {'key': 'recall', 'value': hmean[0] * 100, 'unit': '%', 'display_name': 'Recall'})
    outputs.append(
        {'key': 'precision', 'value': hmean[1] * 100, 'unit': '%', 'display_name': 'Precision'})

    average_precision = collect_ap(os.path.join(work_dir, 'test_py_stdout'))[0]
    outputs.append({'key': 'ap', 'value': average_precision * 100, 'unit': '%',
                    'display_name': 'AP @ [IoU=0.50:0.95]'})
    return outputs


def evaluate(config, snapshot, out, update_config, show_dir):
    """ Main function. """
    logging.basicConfig(level=logging.INFO)

    metrics_functions = (
        (coco_eval, (update_config, show_dir)),
    )
    evaluate_internal(config, snapshot, out, update_config, metrics_functions)
