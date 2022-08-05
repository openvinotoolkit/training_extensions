"""
Install OTE
"""
# Copyright (C) 2022 Intel Corporation
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

from typing import List
from setuptools import find_packages, setup


def get_requirements() -> List[str]:
    """Read requirements.txt and return the requirements.
    Returns:
        List[str]: List of requirements.
    """
    requirements = []
    with open("requirements/ote.txt", "r", encoding="utf8") as file:
        for line in file.readlines():
            requirements.append(line.strip())
    return requirements


setup(
    name="OTE",
    version="2.0.0",
    packages=find_packages(where="ote_sdk", include=["ote_sdk", "ote_sdk.*"])
    + find_packages(where="ote_cli", include=["ote_cli", "ote_cli.*"], exclude=("tools",)),
    package_dir={"ote_sdk": "ote_sdk/ote_sdk", "ote_cli": "ote_cli/ote_cli"},
    url="https://github.com/openvinotoolkit/training_extensions",
    license="license='Apache License 2.0'",
    install_requires=get_requirements(),
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
    author="Intel",
    description="OpenVINO Training Extensions",
)
