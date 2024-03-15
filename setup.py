"""Setup file for OTX."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa

import os
import platform
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import List, Union

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension
from pkg_resources import Requirement
from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension

    cmd_class = {"build_ext_torch_cpp_ext": BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print("Skip building ext ops due to the absence of torch.")

cmd_class["build_ext_cythonized"] = build_ext

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def load_module(name: str = "src/otx/__init__.py"):
    """Load Python Module.

    Args:
        name (str, optional): Name of the module to load.
            Defaults to "src/otx/__init__.py".
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
    otx = load_module(name="src/otx/__init__.py")
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
        with open(f"requirements/{requirement_file}.txt", "r", encoding="UTF-8") as file:
            for line in file:
                package = line.strip()
                if package and not package.startswith(("#", "-f")):
                    Requirement.parse(package)
                    requirements.append(package)

    return requirements


def get_extensions():
    if platform.system() == "Windows":
        return []

    def _cython_modules():
        cython_files = [
            "src/otx/algorithms/common/adapters/mmcv/pipelines/transforms/cython_augments/pil_augment.pyx",
            "src/otx/algorithms/common/adapters/mmcv/pipelines/transforms/cython_augments/cv_augment.pyx"
        ]

        ext_modules = [
            Extension(
                cython_file.rstrip(".pyx").lstrip("src/").replace("/", "."),
                [cython_file],
                include_dirs=[numpy.get_include()],
                extra_compile_args=["-O3"],
            )
            for cython_file in cython_files
        ]

        return cythonize(ext_modules)

    extensions = []
    extensions.extend(_cython_modules())
    return extensions


REQUIRED_PACKAGES = get_requirements(requirement_files="api")
EXTRAS_REQUIRE = {
    "action": get_requirements(requirement_files=[
            "base", "openvino", "action",
        ]
    ),
    "anomaly": get_requirements(requirement_files=[
            "base", "openvino", "anomaly",
        ]
    ),
    "classification": get_requirements(requirement_files=[
            "base", "openvino", "classification",
        ]
    ),
    "detection": get_requirements(requirement_files=[
            "base", "openvino", "detection",
        ]
    ),
    "segmentation": get_requirements(requirement_files=[
            "base", "openvino", "segmentation",
        ]
    ),
    "visual_prompting": get_requirements(requirement_files=[
            "base", "openvino", "visual_prompting",
        ]
    ),
    "full": get_requirements(requirement_files=[
            "base",
            "openvino",
            "anomaly",
            "classification",
            "detection",
            "segmentation",
            "visual_prompting",
            "action",
        ]
    ),
}


def find_yaml_recipes():
    """Find YAML recipe files in the package."""
    results = defaultdict(list)

    for root, _, files in os.walk("src/otx"):
        module = ".".join(root.split(os.sep))
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in [".yaml", ".json"]:
                results[module] += [file]

    return results


package_data = {"": ["README.md", "LICENSE", "py.typed"]}
package_data.update(find_yaml_recipes())

setup(
    name="otx",
    version=get_otx_version(),
    description="OpenVINO™ Training Extensions: "
    "Train, Evaluate, Optimize, Deploy Computer Vision Models via OpenVINO™",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="OpenVINO™ Training Extensions Contributors",
    url="https://github.com/openvinotoolkit/training_extensions",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Cython",
    ],
    license="Apache License 2.0",
    packages=find_packages(where="src", include=["otx*"]),
    package_dir={"": "src"},
    package_data=package_data,
    include_package_data=True,
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
