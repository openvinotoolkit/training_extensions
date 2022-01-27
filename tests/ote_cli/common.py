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

import shutil
import json
import os
from subprocess import run  # nosec

from ote_sdk.usecases.exportable_code.utils import get_git_commit_hash

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
    if 'HTTP_PROXY' in os.environ:
        vars.update({'HTTP_PROXY': os.environ['HTTP_PROXY']})
    if 'HTTPS_PROXY' in os.environ:
        vars.update({'HTTPS_PROXY': os.environ['HTTPS_PROXY']})
    if 'NO_PROXY' in os.environ:
        vars.update({'NO_PROXY': os.environ['NO_PROXY']})
    return vars


def patch_demo_py(src_path, dst_path):
    with open(src_path) as read_file:
        content = [line for line in read_file]
        replaced = False
        for i, line in enumerate(content):
            if 'visualizer = Visualizer(media_type)' in line:
                content[i] = line.rstrip() + '; visualizer.show = show\n'
                replaced = True
        assert replaced
        content = ['def show(self):\n', '    pass\n\n'] + content
        with open(dst_path, 'w') as write_file:
            write_file.write(''.join(content))


def remove_ote_sdk_from_requirements(path):
    with open(path, encoding='UTF-8') as read_file:
        content = ''.join([line for line in read_file if 'ote_sdk' not in line])

    with open(path, 'w', encoding='UTF-8') as write_file:
        write_file.write(content)


def check_ote_sdk_commit_hash_in_requirements(path):
    with open(path, encoding='UTF-8') as read_file:
        content = [line for line in read_file if 'ote_sdk' in line]
    if len(content) != 1:
        raise RuntimeError(f"Invalid ote_sdk requirements (0 or more than 1 times mentioned): {path}")

    git_commit_hash = get_git_commit_hash()
    if git_commit_hash in content[0]:
        return True

    return False


def ote_train_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'train',
                    template.model_template_id,
                    '--train-ann-file',
                    f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                    '--train-data-roots',
                    f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                    '--val-ann-file',
                    f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                    '--val-data-roots',
                    f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                    '--save-model-to',
                    f'{template_work_dir}/trained_{template.model_template_id}']
    command_line.extend(args['train_params'])
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f'{template_work_dir}/trained_{template.model_template_id}/weights.pth')
    assert os.path.exists(f'{template_work_dir}/trained_{template.model_template_id}/label_schema.json')


def ote_hpo_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    if os.path.exists(f"{template_work_dir}/hpo"):
        shutil.rmtree(f"{template_work_dir}/hpo")
    command_line = ['ote',
                    'train',
                    template.model_template_id,
                    '--train-ann-file',
                    f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                    '--train-data-roots',
                    f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                    '--val-ann-file',
                    f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                    '--val-data-roots',
                    f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                    '--save-model-to',
                    f'{template_work_dir}/hpo_trained_{template.model_template_id}',
                    '--enable-hpo',
                    '--hpo-time-ratio',
                    '1']
    command_line.extend(args['train_params'])
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f"{template_work_dir}/hpo/hpopt_status.json")
    with open(f"{template_work_dir}/hpo/hpopt_status.json", "r") as f:
        assert json.load(f).get('best_config_id', None) is not None
    assert os.path.exists(f'{template_work_dir}/hpo_trained_{template.model_template_id}/weights.pth')
    assert os.path.exists(f'{template_work_dir}/hpo_trained_{template.model_template_id}/label_schema.json')


def ote_export_testing(template, root):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'export',
                    template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    f'--save-model-to',
                    f'{template_work_dir}/exported_{template.model_template_id}']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml')
    assert os.path.exists(f'{template_work_dir}/exported_{template.model_template_id}/openvino.bin')
    assert os.path.exists(f'{template_work_dir}/exported_{template.model_template_id}/label_schema.json')


def ote_eval_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'eval',
                    template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    '--save-performance',
                    f'{template_work_dir}/trained_{template.model_template_id}/performance.json']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f'{template_work_dir}/trained_{template.model_template_id}/performance.json')


def ote_eval_openvino_testing(template, root, ote_dir, args, threshold):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'eval',
                    template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    '--save-performance',
                    f'{template_work_dir}/exported_{template.model_template_id}/performance.json']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f'{template_work_dir}/exported_{template.model_template_id}/performance.json')
    with open(f'{template_work_dir}/trained_{template.model_template_id}/performance.json') as read_file:
        trained_performance = json.load(read_file)
    with open(f'{template_work_dir}/exported_{template.model_template_id}/performance.json') as read_file:
        exported_performance = json.load(read_file)

    for k in trained_performance.keys():
        assert abs(trained_performance[k] - exported_performance[k]) / trained_performance[k] <= threshold, f"{trained_performance[k]=}, {exported_performance[k]=}"


def ote_demo_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'demo',
                    template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    '--input',
                    os.path.join(ote_dir, args['--input']),
                    '--delay',
                    '-1']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0


def ote_demo_openvino_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'demo',
                    template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    '--input',
                    os.path.join(ote_dir, args['--input']),
                    '--delay',
                    '-1']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0


def ote_deploy_openvino_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    deployment_dir = f'{template_work_dir}/deployed_{template.model_template_id}'
    command_line = ['ote',
                    'deploy',
                    template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    f'--save-model-to',
                    deployment_dir]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert run(['unzip', 'openvino.zip'],
               cwd=deployment_dir).returncode == 0
    assert run(['python3', '-m', 'venv', 'venv'],
               cwd=os.path.join(deployment_dir, 'python')).returncode == 0
    assert run(['python3', '-m', 'pip', 'install', 'wheel'],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0

    assert check_ote_sdk_commit_hash_in_requirements(os.path.join(deployment_dir, 'python', 'requirements.txt'))

    # Remove ote_sdk from requirements.txt, since merge commit (that is created on CI) is not pushed to github and that's why cannot be cloned.
    # Install ote_sdk from local folder instead.
    # Install the demo_package with --no-deps since, requirements.txt has been embedded to the demo_package during creation.
    remove_ote_sdk_from_requirements(os.path.join(deployment_dir, 'python', 'requirements.txt'))
    assert run(['python3', '-m', 'pip', 'install', '-e', os.path.join(os.path.dirname(__file__), '..', '..', 'ote_sdk')],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0
    assert run(['python3', '-m', 'pip', 'install', '-r', os.path.join(deployment_dir, 'python', 'requirements.txt')],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0
    assert run(['python3', '-m', 'pip', 'install', 'demo_package-0.0-py3-none-any.whl', '--no-deps'],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0

    # Patch demo since we are not able to run cv2.imshow on CI.
    patch_demo_py(os.path.join(deployment_dir, 'python', 'demo.py'),
                  os.path.join(deployment_dir, 'python', 'demo_patched.py'))

    assert run(['python3', 'demo_patched.py', '-m', '../model/model.xml', '-i', os.path.join(ote_dir, args['--input'])],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0


def ote_eval_deployment_testing(template, root, ote_dir, args, threshold):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'eval',
                    template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/deployed_{template.model_template_id}/openvino.zip',
                    '--save-performance',
                    f'{template_work_dir}/deployed_{template.model_template_id}/performance.json']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f'{template_work_dir}/deployed_{template.model_template_id}/performance.json')
    with open(f'{template_work_dir}/exported_{template.model_template_id}/performance.json') as read_file:
        exported_performance = json.load(read_file)
    with open(f'{template_work_dir}/deployed_{template.model_template_id}/performance.json') as read_file:
        deployed_performance = json.load(read_file)

    for k in exported_performance.keys():
        assert abs(exported_performance[k] - deployed_performance[k]) / exported_performance[k] <= threshold, f"{exported_performance[k]=}, {deployed_performance[k]=}"


def ote_demo_deployment_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'demo',
                    template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/deployed_{template.model_template_id}/openvino.zip',
                    '--input',
                    os.path.join(ote_dir, args['--input']),
                    '--delay',
                    '-1']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0