import os
import tempfile
from subprocess import run

venv_to_dir = {
    'models/object_detection' : os.path.join(tempfile.mkdtemp(), 'venv')
}

tests_to_venv = {
    'tests/tasks/test_mmdetection_task.py' : 'models/object_detection',
}

for k, v in venv_to_dir.items():
    outfile = tempfile.mktemp()
    print(f'Initializing environment for {k}')
    res = run(f'cd {k} && ./init_venv.sh {v}', check=True, shell=True)
    print(f'Initializing environment for {k} has bee completed.')
    print('#############################################################################')
    print('#############################################################################')
    print('#############################################################################')
    print('\n\n\n\n\n')

for k, v in tests_to_venv.items():
    run(f'source {venv_to_dir[v]}/bin/activate && pytest {k} -v', check=True, shell=True, executable='/bin/bash')
