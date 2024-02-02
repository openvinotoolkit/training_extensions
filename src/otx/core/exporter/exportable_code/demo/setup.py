# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""setup file for demo package."""

from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIR = Path(__file__).resolve().parent

with Path.open(SETUP_DIR / "requirements.txt", encoding="utf8") as f:
    required = f.read().splitlines()

packages = find_packages(str(SETUP_DIR))
package_dir = {packages[0]: str(SETUP_DIR / packages[0])}

setup(
    name=packages[0],
    version="0.0",
    author="Intel® Corporation",
    license="Copyright (c) 2024 Intel Corporation. SPDX-License-Identifier: Apache-2.0",
    description="Demo based on ModelAPI classes",
    packages=packages,
    package_dir=package_dir,
    package_data={
        packages[0]: ["*.json"],
    },
    install_requires=required,
)
