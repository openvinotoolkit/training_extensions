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

import hashlib
import logging
import os
import requests
import subprocess

import yaml

from tempfile import NamedTemporaryFile

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
            destination_file = os.path.join(output_folder, destination)
            if 'google.com' in source:
                file_id = source.split('id=')[-1]

                session = requests.Session()
                gdrive_url = 'https://docs.google.com/uc?export=download'
                response = session.get(gdrive_url, params={'id': file_id}, stream=True)
                response.raise_for_status()

                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        response = session.get(gdrive_url, params={'id': file_id, 'confirm': value}, stream=True)
                        response.raise_for_status()

                with open(destination_file, 'wb') as f:
                    f.write(response.content)
            else:
                subprocess.run(f'wget -q -O {destination_file} {source}', check=True, shell=True)
            logging.info(f'Downloading {source} has been completed.')

            actual = get_file_size_and_sha256(os.path.join(output_folder, destination))
            assert expected_size == actual['size'], f'{template_file} actual_size {actual["size"]}'
            assert expected_sha256 == actual['sha256'], f'{template_file} actual_sha256 {actual["sha256"]}'

            return

    raise RuntimeError('Failed to find snapshot.pth')

def convert_bash_command_for_log(cmd):
    if not cmd:
        return ''
    if isinstance(cmd, list):
        cmd = ' '.join(cmd)
    cmd = cmd.replace(';', '; ')
    cmd = cmd.split()

    if len(cmd) == 1:
        return cmd[0]

    shift_str = ' ' * 4
    split_str = ' \\\n' + shift_str
    semicolon_split_str = ' \\\n'
    max_line_len = 40
    max_chunk_len_to_keep_line = 30
    min_line_len_to_split_big_chunk = 10

    cur_line = cmd[0]
    cmdstr = ''
    for chunk in cmd[1:]:
        assert chunk
        if len(cur_line) > max_line_len:
            cmdstr += cur_line + split_str
            cur_line = ''
        if cur_line and chunk.startswith('--'):
            cmdstr += cur_line + split_str
            cur_line = ''
        if cur_line and chunk.startswith('|'):
            cmdstr += cur_line + split_str
            cur_line = ''
        if (cur_line
            and len(chunk) > max_chunk_len_to_keep_line
            and len(cur_line) >= min_line_len_to_split_big_chunk):
            cmdstr += cur_line + split_str
            cur_line = ''

        if cur_line:
            cur_line += ' '
        cur_line += chunk

        if cur_line.endswith(';'):
            cmdstr += cur_line + semicolon_split_str
            cur_line = ''

    cmdstr += cur_line

    while cmdstr.endswith(' ') or cmdstr.endswith('\n'):
        cmdstr = cmdstr[:-1]
    return cmdstr

def log_shell_cmd(cmd, prefix='Running through shell cmd'):
    cmdstr = convert_bash_command_for_log(cmd)
    logging.debug(f'{prefix}\n`{cmdstr}\n`')

def run_through_shell(cmd, verbose=True, check=True):
    assert isinstance(cmd, str)
    log_shell_cmd(cmd)
    std_streams_args = {} if verbose else {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
    return subprocess.run(cmd,
                          shell=True,
                          check=check,
                          executable="/bin/bash",
                          **std_streams_args)

def get_cuda_device_count():
    # move `import torch` inside this function to import the function in ote venv
    import torch
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def generate_random_suffix():
    random_suffix = os.path.basename(NamedTemporaryFile().name)
    prefix = 'tmp'
    if random_suffix.startswith(prefix):
        random_suffix = random_suffix[len(prefix):]
    return random_suffix
