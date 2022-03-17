""" Runs tests selectively depending on changed files. """

import os
import sys
from subprocess import run

from tests.ote_cli.common import collect_env_vars

ALGO_ROOT_DIR = "external"
ALGO_DIRS = [
    os.path.join(ALGO_ROOT_DIR, d)
    for d in os.listdir(ALGO_ROOT_DIR)
    if os.path.isdir(os.path.join(ALGO_ROOT_DIR, d))
]
IMPORTANT_DIRS = [
    "data/",
    "ote_cli/",
    "ote_sdk/",
    "tests/",
]

wd = sys.argv[1]


def what_to_test():
    """
    Returns a dict containing information whether it is needed
    to run tests for particular algorithm.
    """

    print(f"{sys.argv=}")
    run_algo_tests = {d: True for d in ALGO_DIRS}
    if len(sys.argv) > 2:
        run_algo_tests = {d: False for d in ALGO_DIRS}
        changed_files = sys.argv[2:]
        print(f"{changed_files=}")

        for changed_file in changed_files:
            if any(changed_file.startswith(d) for d in IMPORTANT_DIRS):
                run_algo_tests = {d: True for d in ALGO_DIRS}
                break

            for d in ALGO_DIRS:
                if changed_file.startswith(d):
                    run_algo_tests[d] = True

    for k, v in run_algo_tests.items():
        print("run", k, v)

    return run_algo_tests


def test(run_algo_tests):
    """
    Runs tests for algorithms and other stuff (misc).
    """

    passed = {}
    success = True
    command = ["pytest", os.path.join("tests", "ote_cli", "misc"), "-v"]
    res = run(command, env=collect_env_vars(wd), check=True).returncode == 0
    passed["misc"] = res
    success *= res
    for algo_dir in ALGO_DIRS:
        if run_algo_tests[algo_dir]:
            command = ["pytest", os.path.join("tests", "ote_cli", algo_dir), "-v", "-s"]
            res = run(command, env=collect_env_vars(wd), check=True).returncode == 0
            passed[algo_dir] = res
            success *= res

    for k, v in passed.items():
        res = "PASSED" if v else "FAILED"
        print(f"Tests for {k} {res}")

    sys.exit(1 - success)


test(what_to_test())
