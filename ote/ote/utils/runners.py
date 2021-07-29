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
import signal
import subprocess
import sys
import time
from queue import Empty, Queue
from threading import Thread

from .misc import log_shell_cmd


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


def run_with_termination(cmd):
    log_shell_cmd(cmd, 'Running with termination the command')
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

    time.sleep(1)
    err = f'Out of memory: Killed process {process.pid}'
    proc = subprocess.Popen(['dmesg', '-l', 'err'], stdout=subprocess.PIPE)
    out = proc.communicate()[0].decode().split('\n')
    for line in out:
        if err in line:
            raise RuntimeError(line)
