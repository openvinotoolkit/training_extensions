"""Setup file for OTE."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def load_module(name: str = "ote/__init__.py"):
    """Load Python Module.

    Args:
        name (str, optional): Name of the module to load.
            Defaults to "ote/__init__.py".
    """
    location = str(Path(__file__).parent / name)
    spec = spec_from_file_location(name=name, location=location)
    module = module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    return module


def get_version() -> str:
    """Get version from `ote.__init__`.

    Version is stored in the main __init__ module in `ote`.
    The varible storing the version is `__version__`. This function
    reads `__init__` file, checks `__version__ variable and return
    the value assigned to it.

    Example:
        >>> # Assume that __version__ = "0.2.6"
        >>> get_version()
        "0.2.6"

    Returns:
        str: `ote` version.
    """
    ote = load_module(name="ote/__init__.py")
    return ote.__version__


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
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        "console_scripts": [
            "ote=ote.cli.tools.ote:main",
            "ote_demo=ote.cli.tools.demo:main",
            "ote_eval=ote.cli.tools.eval:main",
            "ote_export=ote.cli.tools.export:main",
            "ote_find=ote.cli.tools.find:main",
            "ote_train=ote.cli.tools.train:main",
            "ote_optimize=ote.cli.tools.optimize:main",
        ]
    },
)
