"""Setup file for OTE."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List

from setuptools import find_packages, setup


def get_required_packages(requirement_files: List[str]) -> List[str]:
    """Get packages from requirements.txt file.

    This function returns list of required packages from requirement files.

    Args:
        requirement_files (List[str]): txt files that contains list of required
            packages.

    Example:
        >>> get_required_packages(requirement_files=["openvino"])
        ['onnx>=1.8.1', 'networkx~=2.5', 'openvino-dev==2021.4.1', ...]

    Returns:
        List[str]: List of required packages
    """
    required_packages: List[str] = []

    for requirement_file in requirement_files:
        with open(f"requirements/{requirement_file}.txt", "r", encoding="utf8") as file:
            for line in file:
                package = line.strip()
                if package and not package.startswith(("#", "-f")):
                    required_packages.append(package)

    return required_packages


REQUIRED_PACKAGES = get_required_packages(requirement_files=["cli"])

setup(
    name="ote",
    version="0.2",
    packages=find_packages(where="ote", exclude=("tools",)),
    package_dir={"": "ote"},
    install_requires=REQUIRED_PACKAGES,
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
