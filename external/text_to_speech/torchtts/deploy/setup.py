from setuptools import setup, Extension, find_packages
from pathlib import Path



def readme():
    with open('README.md') as f:
        content = f.read()
    return content

def get_requirements():
    requirements = []
    with open('requirements.txt', 'rt') as req_file:
        for line in req_file.readlines():
            line = line.rstrip()
            if line != '':
                requirements.append(line)
    return requirements

SETUP_DIR = Path(__file__).resolve().parent

with open(SETUP_DIR / "requirements.txt", "r", encoding="utf8") as f:
    required = f.read().splitlines()

packages = find_packages(str(SETUP_DIR))
package_dir = {packages[0]: str(SETUP_DIR / packages[0])}

setup(
    name=packages[0],
    version="0.0",
    author="Intel® Corporation",
    license="Copyright (c) 2021-2022 Intel Corporation. "
    "SPDX-License-Identifier: Apache-2.0",
    description="Demo based on ModelAPI classes",
    packages=packages,
    package_dir=package_dir,
    package_data={
        packages[0]: ["*.json"],
    },
    install_requires=required,
)