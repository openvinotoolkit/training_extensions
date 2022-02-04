"""
setup file for OTE SDK
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from setuptools import find_packages, setup

install_requires = []

with open("requirements.txt", "r", encoding="UTF-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            install_requires.append(line)

setup(
    name="OTE SDK",
    version="1.0",
    packages=find_packages(include=["ote_sdk", "ote_sdk.*"]),
    package_data={"ote_sdk": ["py.typed", "usecases/exportable_code/demo/*"]},
    url="",
    license="Copyright (c) 2021-2022 Intel Corporation. "
    "SPDX-License-Identifier: Apache-2.0",
    install_requires=install_requires,
    author="Intel",
    description="OTE SDK Package",
)
