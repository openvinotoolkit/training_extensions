"""Setup file for OTX."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import warnings
from glob import glob
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import List, Optional, Union

import Cython
import numpy
from Cython.Build import cythonize
from pkg_resources import Requirement
from setuptools import Extension, find_packages, setup

try:
    from torch.utils.cpp_extension import CppExtension, BuildExtension

    cmd_class = {"build_ext": BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print("Skip building ext ops due to the absence of torch.")


def load_module(name: str = "otx/__init__.py"):
    """Load Python Module.

    Args:
        name (str, optional): Name of the module to load.
            Defaults to "otx/__init__.py".
    """
    location = str(Path(__file__).parent / name)
    spec = spec_from_file_location(name=name, location=location)
    module = module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    return module


def get_otx_version() -> str:
    """Get version from `otx.__init__`.

    Version is stored in the main __init__ module in `otx`.
    The varible storing the version is `__version__`. This function
    reads `__init__` file, checks `__version__ variable and return
    the value assigned to it.

    Example:
        >>> # Assume that __version__ = "0.2.6"
        >>> get_otx_version()
        "0.2.6"

    Returns:
        str: `otx` version.
    """
    otx = load_module(name="otx/__init__.py")
    return otx.__version__


def get_requirements(requirement_files: Union[str, List[str]]) -> List[str]:
    """Get packages from requirements.txt file.

    This function returns list of required packages from requirement files.

    Args:
        requirement_files (List[str]): txt files that contains list of required
            packages.

    Example:
        >>> get_requirements(requirement_files=["openvino"])
        ['onnx>=1.8.1', 'networkx~=2.5', 'openvino-dev==2021.4.1', ...]

    Returns:
        List[str]: List of required packages
    """
    if isinstance(requirement_files, str):
        requirement_files = [requirement_files]

    requirements: List[str] = []
    for requirement_file in requirement_files:
        with open(
            f"requirements/{requirement_file}.txt", "r", encoding="UTF-8"
        ) as file:
            for line in file:
                package = line.strip()
                if package and not package.startswith(("#", "-f")):
                    requirement = Requirement.parse(package)
                    requirements.append(package)

    return requirements


def get_extensions():
    def _cython_modules():
        package_root = os.path.dirname(__file__)

        cython_files = [
            "otx/mpa/modules/datasets/pipelines/transforms/cython_augments/pil_augment.pyx",
            "otx/mpa/modules/datasets/pipelines/transforms/cython_augments/cv_augment.pyx",
        ]

        ext_modules = [
            Extension(
                cython_file.rstrip(".pyx").replace("/", "."),
                [os.path.join(package_root, cython_file)],
                include_dirs=[numpy.get_include()],
                extra_compile_args=["-O3"],
            )
            for cython_file in cython_files
        ]

        return cythonize(ext_modules, annotate=True)

    def _torch_modules():
        ext_modules = []

        # prevent ninja from using too many resources
        os.environ.setdefault("MAX_JOBS", "4")
        extra_compile_args = {"cxx": []}

        # otx.mpa.modules._mpl
        op_files = glob("./otx/mpa/csrc/mpl/*.cpp")
        include_path = os.path.abspath("./otx/mpa/csrc/mpl")
        ext_ops = CppExtension(
            name="otx.mpa.modules._mpl",
            sources=op_files,
            include_dirs=[include_path],
            define_macros=[],
            extra_compile_args=extra_compile_args,
        )
        ext_modules.append(ext_ops)
        return ext_modules

    extensions = []

    extensions.extend(_torch_modules())
    extensions.extend(_cython_modules())

    return extensions


REQUIRED_PACKAGES = get_requirements(requirement_files=["base", "dev", "openvino"])
EXTRAS_REQUIRE = {
    "action": get_requirements(requirement_files="action"),
    "anomaly": get_requirements(requirement_files="anomaly"),
    "classification": get_requirements(requirement_files="classification"),
    "detection": get_requirements(requirement_files="detection"),
    "segmentation": get_requirements(requirement_files="segmentation"),
    "mpa": get_requirements(
        requirement_files=["classification", "detection", "segmentation", "action"]
    ),
    "full": get_requirements(
        requirement_files=[
            "anomaly",
            "classification",
            "detection",
            "segmentation",
            "action",
        ]
    ),
}


setup(
    name="otx",
    version=get_otx_version(),
    packages=find_packages(exclude=("tests",)),
    package_data={
        "": ["requirements.txt", "README.md", "LICENSE"]
    },  # Needed for exportable code
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "otx=otx.cli.tools.cli:main",
            "otx_demo=otx.cli.tools.demo:main",
            "otx_eval=otx.cli.tools.eval:main",
            "otx_export=otx.cli.tools.export:main",
            "otx_find=otx.cli.tools.find:main",
            "otx_train=otx.cli.tools.train:main",
            "otx_optimize=otx.cli.tools.optimize:main",
            "otx_build=otx.cli.tools.build:main",
        ]
    },
)
