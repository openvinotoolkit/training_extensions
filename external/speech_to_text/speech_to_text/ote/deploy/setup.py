# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIR = Path(__file__).resolve().parent

with open(SETUP_DIR / "requirements.txt", "r", encoding="utf8") as f:
    required = f.read().splitlines()

packages = find_packages(str(SETUP_DIR))
package_dir = {packages[0]: str(SETUP_DIR / packages[0])}

setup(
    name=packages[0],
    version="0.0",
    author="Intel® Corporation",
    license="Apache-2.0",
    description="OTE Speech to Text Demo",
    packages=packages,
    package_dir=package_dir,
    package_data={
        packages[0]: ["*.json", "data/*"],
    },
    install_requires=required
)
