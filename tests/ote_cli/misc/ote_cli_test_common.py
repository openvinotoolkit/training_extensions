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
from subprocess import run  # nosec
import logging


default_train_args_paths = {
    '--train-ann-file': 'data/airport/annotation_example_train.json',
    '--train-data-roots': 'data/airport/train',
    '--val-ann-file': 'data/airport/annotation_example_train.json',
    '--val-data-roots': 'data/airport/train',
    '--test-ann-files': 'data/airport/annotation_example_train.json',
    '--test-data-roots': 'data/airport/train',
}

wrong_paths = {
               'empty': '',
               'not_printable': '\x11',
               # 'null_symbol': '\x00' It is caught on subprocess level
               }

logger = logging.getLogger(__name__)


def get_template_rel_dir(template):
    return os.path.dirname(os.path.relpath(template.model_template_path))


def get_some_vars(template, root):
    template_dir = get_template_rel_dir(template)
    task_type = template.task_type
    work_dir = os.path.join(root, str(task_type))
    template_work_dir = os.path.join(work_dir, template_dir)
    os.makedirs(template_work_dir, exist_ok=True)
    algo_backend_dir = '/'.join(template_dir.split('/')[:2])

    return work_dir, template_work_dir, algo_backend_dir


def create_venv(algo_backend_dir, work_dir, template_work_dir):
    venv_dir = f'{work_dir}/venv'
    if not os.path.exists(venv_dir):
        assert run([f'./{algo_backend_dir}/init_venv.sh', venv_dir]).returncode == 0
        assert run([f'{work_dir}/venv/bin/python', '-m', 'pip', 'install', '-e', 'ote_cli']).returncode == 0


def extract_export_vars(path):
    inner_vars = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('export ') and '=' in line:
                line = line.replace('export ', '').split('=')
                assert len(line) == 2
                inner_vars[line[0].strip()] = line[1].strip()
    return inner_vars


def collect_env_vars(work_dir):
    inner_vars = extract_export_vars(f'{work_dir}/venv/bin/activate')
    inner_vars.update({'PATH':f'{work_dir}/venv/bin/:' + os.environ['PATH']})
    if 'HTTP_PROXY' in os.environ:
        inner_vars.update({'HTTP_PROXY': os.environ['HTTP_PROXY']})
    if 'HTTPS_PROXY' in os.environ:
        inner_vars.update({'HTTPS_PROXY': os.environ['HTTPS_PROXY']})
    if 'NO_PROXY' in os.environ:
        inner_vars.update({'NO_PROXY': os.environ['NO_PROXY']})
    return inner_vars


def ote_common(template, root, tool, cmd_args):
    work_dir, __, _ = get_some_vars(template, root)
    command_line = ['ote',
                    tool,
                    *cmd_args]
    ret = run(command_line, env=collect_env_vars(work_dir), capture_output=True)
    output = {'exit_code': int(ret.returncode), 'stdout': str(ret.stdout), 'stderr': str(ret.stderr)}
    logger.debug(f"Command arguments: {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {output['stdout']}\n")
    logger.debug(f"Stderr: {output['stderr']}\n")
    logger.debug(f"Exit_code: {output['exit_code']}\n")
    return output


def get_pretrained_artifacts(template, root, ote_dir):
    _, template_work_dir, _ = get_some_vars(template, root)
    pretrained_artifact_path = f"{template_work_dir}/trained_{template.model_template_id}"
    logger.debug(f">>> Current pre-trained artifact: {pretrained_artifact_path}")
    command_args = [
        template.model_template_id,
        "--train-ann-file",
        f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
        "--train-data-roots",
        f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
        "--val-ann-file",
        f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
        "--val-data-roots",
        f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
        "--save-model-to",
        pretrained_artifact_path,
    ]
    if not os.path.exists(pretrained_artifact_path):
        ote_common(template, root, 'train', command_args)
        assert os.path.exists(pretrained_artifact_path), f"The folder must exists after command execution"
        weights = os.path.join(pretrained_artifact_path, 'weights.pth')
        labels = os.path.join(pretrained_artifact_path, 'label_schema.json')
        assert os.path.exists(weights), f"The {weights} must exists after command execution"
        assert os.path.exists(labels), f"The {labels} must exists after command execution"
