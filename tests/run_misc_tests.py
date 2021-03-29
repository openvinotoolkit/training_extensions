import os
import tempfile
from subprocess import run

temp_dir = tempfile.mkdtemp()

wd = os.path.join('misc', 'pytorch_toolkit')
for directory in os.listdir(wd):
    path = os.path.join(wd, directory)
    if os.path.isdir(path):
        if {'tests', 'init_venv.sh'}.issubset(set(os.listdir(path))):
            run(
                f'cd {path} && '
                f' ./init_venv.sh {os.path.join(temp_dir, directory)} &&'
                f' source {os.path.join(temp_dir, directory)}/bin/activate &&'
                ' pip install pytest &&'
                ' pytest tests/',
                check=True,
                shell=True,
                executable="/bin/bash",
            )
