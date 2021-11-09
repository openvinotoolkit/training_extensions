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