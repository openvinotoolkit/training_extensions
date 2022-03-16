import os
import re
import sys
from subprocess import run

from tests.ote_cli.common import collect_env_vars

algo_root_dir = "external"
algo_dirs = [
    os.path.join(algo_root_dir, dir)
    for dir in os.listdir(algo_root_dir)
    if os.path.isdir(os.path.join(algo_root_dir, dir))
]
run_all_if_changed = [
    "data/",
    "ote_cli/",
    "ote_sdk/",
    "tests/",
]

run_algo_tests = {algo_dir: False for algo_dir in algo_dirs}

run_all = True
wd = sys.argv[1]

print(f"{sys.argv=}")
if len(sys.argv) > 2:
    run_all = False
    changed_files = sys.argv[2:]
    print(f"{changed_files=}")
    for changed_file in changed_files:
        print(f"{changed_file=}")
        for significant_change in run_all_if_changed:
            if changed_file.startswith(significant_change):
                print(f"{significant_change=}")
                run_all = True
                break
        for algo_dir in algo_dirs:
            if changed_file.startswith(algo_dir):
                run_algo_tests[algo_dir] = True

print(f"{run_all=}")
for k, v in run_algo_tests.items():
    print("run", k, v)

passed = {}
success = True
command = ["pytest", os.path.join("tests", "ote_cli", "misc"), "-v"]
res = run(command, env=collect_env_vars(wd)).returncode == 0
passed["misc"] = res
success *= res
for algo_dir in algo_dirs:
    if run_all or run_algo_tests[algo_dir]:
        command = ["pytest", os.path.join("tests", "ote_cli", algo_dir), "-v"]
        res = run(command, env=collect_env_vars(wd)).returncode == 0
        passed[algo_dir] = res
        success *= res

for k, v in passed.items():
    res = "PASSED" if v else "FAILED"
    print(f"Tests for {k} {res}")

sys.exit(1 - success)
