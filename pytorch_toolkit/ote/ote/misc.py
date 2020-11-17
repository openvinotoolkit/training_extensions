import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from queue import Queue, Empty
from threading import Thread

import yaml
from ote import MMDETECTION_TOOLS


def sha256sum(filename):
    """ Computes sha256sum. """

    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def get_file_size_and_sha256(snapshot):
    """ Gets size and sha256 of a file. """

    return {
        'sha256': sha256sum(snapshot),
        'size': os.path.getsize(snapshot),
        'name': os.path.basename(snapshot),
        'source': snapshot
    }


class NonBlockingStreamReader:

    def __init__(self, stream):
        self.stream = stream
        self.queue = Queue()

        def populate_queue(stream, queue):
            while True:
                line = stream.readline()
                if line:
                    queue.put(line)
                else:
                    time.sleep(1)

        self.thread = Thread(target=populate_queue, args=(self.stream, self.queue))
        self.thread.daemon = True
        self.thread.start()

    def readline(self, timeout=None):
        try:
            return self.queue.get(block=timeout is not None, timeout=timeout)
        except Empty:
            return None


def get_complexity_and_size(cfg, config_path, work_dir, update_config, complexity_img_shape=None):
    """ Gets complexity and size of a model. """

    if complexity_img_shape is None:
        image_shape = [x['img_scale'] for x in cfg.test_pipeline if 'img_scale' in x][0][::-1]
        image_shape = " ".join([str(x) for x in image_shape])
    else:
        image_shape = complexity_img_shape

    res_complexity = os.path.join(work_dir, "complexity.json")
    update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
    update_config = f' --update_config {update_config}' if update_config else ''
    subprocess.run(
        f'python {MMDETECTION_TOOLS}/get_flops.py'
        f' {config_path}'
        f' --shape {image_shape}'
        f' --out {res_complexity}'
        f'{update_config}'.split(' '), check=True)
    with open(res_complexity) as read_file:
        content = json.load(read_file)
    return content


def run_with_termination(cmd):
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE)

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

    while True:
        stderr = nbsr_err.readline(0.1)
        if not stderr:
            break
        stderr = stderr.decode('utf-8')
        print(stderr, end='')
        sys.stdout.flush()


def get_work_dir(cfg, update_config):
    overridden_work_dir = update_config.get('work_dir', None)
    return overridden_work_dir[0][1] if overridden_work_dir else cfg.work_dir


def download_snapshot_if_not_yet(template_file, output_folder):
    with open(template_file) as read_file:
        content = yaml.load(read_file, yaml.SafeLoader)

    for dependency in content['dependencies']:
        destination = dependency['destination']
        if destination == 'snapshot.pth':
            source = dependency['source']
            expected_size = dependency['size']
            expected_sha256 = dependency['sha256']
            if os.path.exists(os.path.join(output_folder, destination)):
                actual = get_file_size_and_sha256(os.path.join(output_folder, destination))
                if expected_size == actual['size'] and expected_sha256 == actual['sha256']:
                    logging.info(f'{source} has been already downloaded.')
                    return

            logging.info(f'Downloading {source}')
            subprocess.run(f'wget -q -O {os.path.join(output_folder, destination)} {source}', check=True, shell=True)
            logging.info(f'Downloading {source} has been completed.')
            actual = get_file_size_and_sha256(os.path.join(output_folder, destination))
            assert expected_size == actual['size'], f'{template_file} actual_size {actual["size"]}'
            assert expected_sha256 == actual['sha256'], f'{template_file} actual_sha256 {actual["sha256"]}'
            return

    raise RuntimeError('Failed to find snapshot.pth')
