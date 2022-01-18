"""
Utils for exportable code.
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import subprocess


def get_git_commit_hash() -> str:
    """
    Retuns current git commit hash of OTE.
    """

    work_dir = os.path.dirname(os.path.abspath(__file__))
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work_dir)
        .decode("ascii")
        .strip()
    )


def set_proper_git_commit_hash(path: str) -> None:
    """
    Replaces OTE_COMMIT by OTE git commit hash in a file.
    """

    with open(path, encoding="UTF-8") as read_file:
        content = "".join(read_file)
    to_replace = "OTE_COMMIT"
    if to_replace not in content:
        raise RuntimeError(f"There is no {to_replace} in {path}")
    content = content.replace(to_replace, get_git_commit_hash())
    with open(path, "w", encoding="UTF-8") as write_file:
        write_file.write(content)
