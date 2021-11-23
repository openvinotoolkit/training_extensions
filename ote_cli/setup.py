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

from setuptools import find_packages, setup

with open(
    os.path.join(os.path.dirname(__file__), "requirements.txt"),
    encoding="UTF-8",
) as read_file:
    requirements = [requirement.strip() for requirement in read_file]

setup(
    name="ote_cli",
    version="0.2",
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
        ]
    },
)
