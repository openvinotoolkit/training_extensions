# Copyright (C) 2021 Intel Corporation
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

import os
from subprocess import run

def get_template_rel_dir(template):
    return os.path.dirname(os.path.relpath(template['path']))


def get_some_vars(template, root):
    template_dir = get_template_rel_dir(template)
    task_type = template['task_type']
    work_dir = os.path.join(root, task_type)
    template_work_dir = os.path.join(work_dir, template_dir)
    algo_backend_dir = '/'.join(template_dir.split('/')[:2])

    return template_dir, work_dir, template_work_dir, algo_backend_dir


def create_venv(algo_backend_dir, work_dir, template_work_dir):
    venv_dir = f'{work_dir}/venv'
    if not os.path.exists(venv_dir):
        assert run([f'./{algo_backend_dir}/init_venv.sh', venv_dir]).returncode == 0
        assert run([f'{work_dir}/venv/bin/python', '-m', 'pip', 'install', '-e', 'ote_cli']).returncode == 0
    os.makedirs(template_work_dir, exist_ok=True)


def extract_export_vars(path):
    vars = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('export ') and '=' in line:
                line = line.replace('export ', '').split('=')
                assert len(line) == 2
                vars[line[0].strip()] = line[1].strip()
    return vars


def collect_env_vars(work_dir):
    vars = extract_export_vars(f'{work_dir}/venv/bin/activate')
    vars.update({'PATH':f'{work_dir}/venv/bin/:' + os.environ['PATH']})
    print(f'{vars=}')
    return vars
