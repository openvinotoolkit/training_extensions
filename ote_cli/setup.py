"""
Setup configuration.
"""

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
import re

from setuptools import find_packages, setup

def find_version():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(project_dir, "ote_cli", "version.py")

    version_text = None
    with open(file_path, "r") as version_file:
        lines = version_file.readlines()
        for line in lines:
            if "VERSION = " in line:
                version_text = line

    if version_text is None:
        raise RuntimeError(f"Failed to find version string 'VERSION = ' in '{file_path}'")

    # PEP440:
    # https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
    pep_regex = r"([1-9]\d*!)?(0|[1-9]\d*)(\.(0|[1-9]\d*))*((a|b|rc)(0|[1-9]\d*))?(\.post(0|[1-9]\d*))?(\.dev(0|[1-9]\d*))?"
    version_regex = r"VERSION\s*=\s*.(" + pep_regex + ")."
    match = re.match(version_regex, version_text)
    if not match:
        raise RuntimeError(f"Failed to find version string in '{file_path}'")

    version = version_text[match.start(1) : match.end(1)]
    return version

with open(
    os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="UTF-8"
) as read_file:
    requirements = [requirement.strip() for requirement in read_file]

setup(
    name="ote_cli",
    version=find_version(),
    packages=find_packages(exclude=("tools",)),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ote=ote_cli.tools.ote:main",
            "ote_demo=ote_cli.tools.demo:main",
            "ote_eval=ote_cli.tools.eval:main",
            "ote_export=ote_cli.tools.export:main",
            "ote_find=ote_cli.tools.find:main",
            "ote_train=ote_cli.tools.train:main",
            "ote_optimize=ote_cli.tools.optimize:main",
        ]
    },
)
