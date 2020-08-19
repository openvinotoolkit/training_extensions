import logging
import os
import subprocess
import tempfile

import yaml
from mmcv import Config
from oteod import MMDETECTION_TOOLS
from oteod.misc import get_complexity_and_size, get_file_size_and_sha256


def collect_ap(path):
    """ Collects average precision values in log file. """

    average_precisions = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        for line in content:
            if line.startswith(beginning):
                average_precisions.append(float(line.replace(beginning, '')))
    return average_precisions


def coco_ap_eval(config_path, work_dir, snapshot, outputs, update_config, show_dir=''):
    """ Computes COCO AP. """
    assert isinstance(update_config, dict)

    try:
        res_pkl = os.path.join(work_dir, 'res.pkl')
        with open(os.path.join(work_dir, 'test_py_stdout'), 'w') as test_py_stdout:

            update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
            update_config = f' --update_config {update_config}' if update_config else ''
            show_dir = f' --show-dir {show_dir}' if show_dir else ''
            subprocess.run(
                f'python {MMDETECTION_TOOLS}/test.py'
                f' {config_path} {snapshot}'
                f' --out {res_pkl} --eval bbox'
                f'{show_dir}{update_config}'.split(' '), stdout=test_py_stdout,
                check=True)
        average_precision = collect_ap(os.path.join(work_dir, 'test_py_stdout'))[0]
        outputs.append({'key': 'ap', 'value': average_precision * 100, 'unit': '%',
                        'display_name': 'AP @ [IoU=0.50:0.95]'})
    except:
        outputs.append(
            {'key': 'ap', 'value': None, 'unit': '%', 'display_name': 'AP @ [IoU=0.50:0.95]'})
    with open(os.path.join(work_dir, 'test_py_stdout')) as test_py_stdout:
        logging.info(''.join(test_py_stdout.readlines()))
    return outputs


def evaluate_internal(config_path, snapshot, out, update_config, metrics_functions):
    """ Main evaluation procedure. """

    assert isinstance(update_config, dict)

    cfg = Config.fromfile(config_path)

    work_dir = tempfile.mkdtemp()
    print('results are stored in:', work_dir)

    if os.path.islink(snapshot):
        snapshot = os.path.join(os.path.dirname(snapshot), os.readlink(snapshot))

    files = get_file_size_and_sha256(snapshot)

    metrics = []

    metrics = get_complexity_and_size(cfg, config_path, work_dir, metrics, update_config)
    for func, args in metrics_functions:
        metrics = func(config_path, work_dir, snapshot, metrics, *args)

    for metric in metrics:
        metric['value'] = round(metric['value'], 3) if metric['value'] else metric['value']

    outputs = {
        'files': [files],
        'metrics': metrics
    }

    if os.path.exists(out):
        with open(out) as read_file:
            content = yaml.load(read_file, Loader=yaml.FullLoader)
        content.update(outputs)
        outputs = content

    with open(out, 'w') as write_file:
        yaml.dump(outputs, write_file)


def evaluate(config, snapshot, out, update_config, show_dir):
    """ Main function. """
    logging.basicConfig(level=logging.INFO)

    assert isinstance(update_config, dict)

    metrics_functions = (
        (coco_ap_eval, (update_config, show_dir)),
    )
    evaluate_internal(config, snapshot, out, update_config, metrics_functions)
