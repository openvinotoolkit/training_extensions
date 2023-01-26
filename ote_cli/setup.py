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
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from setuptools import find_packages, setup


def load_module(name: str = "ote_cli/__init__.py"):
    """Load Python Module.

    Args:
        name (str, optional): Name of the module to load.
            Defaults to "ote_cli/__init__.py".
    """
    location = str(Path(__file__).parent / name)
    spec = spec_from_file_location(name=name, location=location)
    module = module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    return module


def get_ote_cli_version() -> str:
    """Get version from `ote_cli.__init__`.

    Version is stored in the main __init__ module in `ote_cli`.
    The varible storing the version is `__version__`. This function
    reads `__init__` file, checks `__version__ variable and return
    the value assigned to it.

    Example:
        >>> # Assume that __version__ = "0.2.6"
        >>> get_ote_cli_version()
        "0.2.6"

    Returns:
        str: `ote_cli` version.
    """
    ote_cli = load_module(name="ote_cli/__init__.py")
    return ote_cli.__version__


with open(
    os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="UTF-8"
) as read_file:
    requirements = [requirement.strip() for requirement in read_file]

setup(
    name="ote_cli",
    version=get_ote_cli_version(),
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
