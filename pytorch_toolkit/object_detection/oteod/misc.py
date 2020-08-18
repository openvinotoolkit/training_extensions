# pylint: disable=W0702,W1203,R0913

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from queue import Queue, Empty
from threading import Thread

from oteod import MMDETECTION_TOOLS

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


def sha256sum(filename):
    """ Computes sha256sum. """

    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


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


def get_work_dir(cfg, update_config):
    overridden_work_dir = [p.split('=') for p in update_config.strip().split(' ') if
                           p.startswith('work_dir=')]
    return overridden_work_dir[0][1] if overridden_work_dir else cfg.work_dir


