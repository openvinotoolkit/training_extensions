# pylint: disable=W0702

import hashlib
import json
import logging
import os
from queue import Queue, Empty
import signal
import subprocess
import sys
import tempfile
from threading import Thread
import time
from mmcv import Config
import yaml
import torch


class NonBlockingStreamReader:

    def __init__(self, stream):
        self.stream = stream
        self.queue = Queue()

        def populate_queue(stream, queue):
            while True:
                line = stream.readline()
                if line:
                    queue.put(line)

        self.thread = Thread(target=populate_queue, args=(self.stream, self.queue))
        self.thread.daemon = True
        self.thread.start()

    def readline(self, timeout=None):
        try:
            return self.queue.get(block=timeout is not None, timeout=timeout)
        except Empty:
            return None


MMDETECTION_TOOLS = f'{os.path.dirname(__file__)}/../../../external/mmdetection/tools'


def replace_text_in_file(path, replace_what, replace_by):
    """ Replaces text in file. """

    with open(path) as read_file:
        content = '\n'.join([line.rstrip() for line in read_file.readlines()])
        if content.find(replace_what) == -1:
            return False
        content = content.replace(replace_what, replace_by)
    with open(path, 'w') as write_file:
        write_file.write(content)
    return True


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


def sha256sum(filename):
    """ Computes sha256sum. """

    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def coco_ap_eval(config_path, work_dir, snapshot, outputs, update_config, show_dir=''):
    """ Computes COCO AP. """
    try:
        res_pkl = os.path.join(work_dir, 'res.pkl')
        with open(os.path.join(work_dir, 'test_py_stdout'), 'w') as test_py_stdout:
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
        print(''.join(test_py_stdout.readlines()))
        sys.stdout.flush()
    return outputs


def get_complexity_and_size(cfg, config_path, work_dir, outputs, update_config):
    """ Gets complexity and size of a model. """

    image_shape = [x['img_scale'] for x in cfg.test_pipeline if 'img_scale' in x][0][::-1]
    image_shape = " ".join([str(x) for x in image_shape])

    res_complexity = os.path.join(work_dir, "complexity.json")
    update_config = f' --update_config {update_config}' if update_config else ''
    subprocess.run(
        f'python {MMDETECTION_TOOLS}/get_flops.py'
        f' {config_path}'
        f' --shape {image_shape}'
        f' --out {res_complexity}'
        f'{update_config}'.split(' '), check=True)
    with open(res_complexity) as read_file:
        content = json.load(read_file)
        outputs.extend(content)
    return outputs


def get_file_size_and_sha256(snapshot):
    """ Gets size and sha256 of a file. """

    return {
        'sha256': sha256sum(snapshot),
        'size': os.path.getsize(snapshot),
        'name': os.path.basename(snapshot),
        'source': snapshot
    }


def evaluate(config_path, snapshot, out, update_config, metrics_functions):
    """ Main evaluation procedure. """

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


def run_with_termination(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    nbsr_err = NonBlockingStreamReader(process.stderr)

    failure_word = 'CUDA out of memory'
    while process.poll() is None:
        stderr = nbsr_err.readline(0.1)
        if stderr is None:
            time.sleep(1)
            continue
        stderr = stderr.decode('utf-8')
        print(stderr, end='')
        sys.stdout.flush()
        if failure_word in stderr:
            try:
                print('\nTerminated because of:', failure_word)
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError as e:
                print(e)


def get_work_dir(cfg, update_config):
    overridden_work_dir = [p.split('=') for p in update_config.strip().split(' ') if
                           p.startswith('work_dir=')]
    return overridden_work_dir[0][1] if overridden_work_dir else cfg.work_dir


def train(config, gpu_num, update_config):
    training_info = {'training_gpu_num': 0}
    if torch.cuda.is_available():
        logging.info('Training on GPUs started ...')
        available_gpu_num = torch.cuda.device_count()
        if available_gpu_num < gpu_num:
            logging.warning(f'available_gpu_num < args.gpu_num: {available_gpu_num} < {gpu_num}')
            logging.warning(f'decreased number of gpu to: {available_gpu_num}')
            gpu_num = available_gpu_num
            sys.stdout.flush()
        run_with_termination(f'{MMDETECTION_TOOLS}/dist_train.sh'
                             f' {config}'
                             f' {gpu_num}'
                             f'{update_config}'.split(' '))
        training_info['training_gpu_num'] = gpu_num
        logging.info('... training on GPUs completed.')
    else:
        logging.info('Training on CPU started ...')
        run_with_termination(f'python {MMDETECTION_TOOLS}/train.py'
                             f' {config}'
                             f'{update_config}'.split(' '))
        logging.info('... training on CPU completed.')

    return training_info
