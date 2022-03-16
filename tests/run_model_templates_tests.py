from pickle import TRUE
from sre_constants import SUCCESS
import sys
import os
from subprocess import run
from tests.ote_cli.common import collect_env_vars


algo_root_dir = "external"
algo_dirs = [os.path.join(algo_root_dir, dir) for dir in os.listdir(algo_root_dir) if os.path.isdir(os.path.join(algo_root_dir, dir))]
run_all_if_changed = [
    "data/",
    "ote_cli/",
    "ote_sdk/",
    "tests/",
]

run_algo_tests = {algo_dir:False for algo_dir in algo_dirs}

run_all = True
wd = sys.argv[1]
if len(sys.argv) > 2:
    run_all = False
    changed_files = sys.argv[2:]
    for changed_file in changed_files:
        print(changed_file)
        for significant_change in run_all_if_changed:
            if changed_file.startswith(significant_change):
                run_all = True
                break
        for algo_dir in algo_dirs:
            if changed_file.startswith(algo_dir):
                run_algo_tests[algo_dir] = True

success = True
command = ["pytest", os.path.join("tests", "ote_cli", "misc"), "-v"]
success *= run(command, env=collect_env_vars(wd)).returncode == 0
for algo_dir in algo_dirs:
    if run_all or run_algo_tests[algo_dir]:
        command = ["pytest", os.path.join("tests", "ote_cli", algo_dir), "-v"]
        success *= run(command, env=collect_env_vars(wd)).returncode == 0

sys.exit(1 - success)
