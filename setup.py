"""Setup file for OTX."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import List, Optional, Union

from pkg_resources import Requirement
from setuptools import find_packages, setup


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
        >>> get_version()
        "0.2.6"

    Returns:
        str: `otx` version.
    """
    otx = load_module(name="otx/__init__.py")
    return otx.__version__


def get_cuda_version() -> Optional[str]:
    """Get CUDA version.

    Output of nvcc --version command would be as follows:

    b'nvcc: NVIDIA (R) Cuda compiler driver\n
    Copyright (c) 2005-2021 NVIDIA Corporation\n
    Built on Sun_Aug_15_21:14:11_PDT_2021\n
    Cuda compilation tools, release 11.4, V11.4.120\n
    Build cuda_11.4.r11.4/compiler.30300941_0\n'

    It would be possible to get the version by getting
    the version number after "release" word in the string.
    """
    cuda_version: Optional[str] = None

    try:
        # 1. Try with nvcc
        output = subprocess.check_output("nvcc --version", shell=True)
        cuda_version = str(output).split("release ")[-1][:4]
    except subprocess.CalledProcessError:
        # 2. Try with CUDA_HOME/version.txt
        cuda_home_path = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        cuda_version_file = os.path.join(cuda_home_path, "version.txt")
        if os.path.exists(cuda_version_file):
            with open(cuda_version_file, "r", encoding="UTF-8") as f:
                cuda_version = f.readline().strip().split()[-1]
        else:
            warnings.warn(f"CUDA is not installed on this {sys.platform} machine.")
    return cuda_version


def update_version_with_cuda_suffix(name: str, version: str) -> str:
    """Update the requirement version with the correct CUDA suffix.

    This function checks whether CUDA is installed on the system. If yes, it
    adds finds the torch cuda suffix to properly install the right version of
    torch or torchvision.

    Args:
        name (str): Name of the requirement. (torch or torchvision.)
        version (str): Version of the requirement.

    Examples:
        Below examples demonstrate how the function append the cuda version.
        Note that the appended cuda version might be different on your system.

        >>> update_version_with_cuda_suffix("torch", "1.8.1")
        '1.8.1+cu111'

        >>> update_version_with_cuda_suffix("torch", "1.12.1")
        '1.12.1+cu113'

    Returns:
        str: Updated version with the correct cuda suffix.
    """

    # version[cuda]: suffix.
    # For example torch 1.8.0 Cuda 10., suffix would be 102.
    supported_torch_cuda_versions = {
        "torch": {
            "1.8.0": {"10": "102", "11": "111"},
            "1.8.1": {"10": "102", "11": "111"},
            "1.9.0": {"10": "102", "11": "111"},
            "1.9.1": {"10": "102", "11": "111"},
            "1.10.0": {"10": "102", "11": "113"},
            "1.10.1": {"10": "102", "11": "113"},
            "1.11.0": {"10": "102", "11": "113"},
            "1.12.0": {"10": "102", "11": "113"},
            "1.12.1": {"10": "102", "11": "113"},
        },
        "torchvision": {
            "0.9.0": {"10": "102", "11": "111"},
            "0.9.1": {"10": "102", "11": "111"},
            "0.10.0": {"10": "102", "11": "111"},
            "0.10.1": {"10": "102", "11": "111"},
            "0.11.0": {"10": "102", "11": "113"},
            "0.11.2": {"10": "102", "11": "113"},
            "0.12.0": {"10": "102", "11": "113"},
            "1.12.0": {"10": "102", "11": "113"},
        },
    }

    suffix: str = ""
    if sys.platform in ["linux", "win32"]:
        # ``get_cuda version()`` returns the exact version such as 11.2. Here
        # we only need the major CUDA version such as 10 or 11, not the minor
        # version. That's why we use [:2] to get the major version.
        cuda = get_cuda_version()
        if cuda is not None:
            suffix = f"+cu{supported_torch_cuda_versions[name][version][cuda[:2]]}"
        else:
            suffix = "+cpu"

    return f"{version}{suffix}"


def update_torch_requirement(requirement: Requirement) -> str:
    """Update torch requirement with the corrected cuda suffix.

    Args:
        requirement (Requirement): Requirement object comprising requirement
            details.

    Examples:
        >>> from pkg_resources import Requirement
        >>> req = "torch>=1.8.1, <=1.9.1"
        >>> requirement = Requirement.parse(req)
        >>> requirement.name
        'torch'
        >>> requirement.specs
        [('>=', '1.8.1'), ('<=', '1.9.1')]
        >>> update_torch_requirement(requirement)
        'torch<=1.9.1+cu111, >=1.8.1+cu111'

        >>> from pkg_resources import Requirement
        >>> req = "torch>=1.8.1"
        >>> requirement = Requirement.parse(req)
        >>> requirement.name
        'torch'
        >>> requirement.specs
        [('>=', '1.8.1')]
        >>> update_torch_requirement(requirement)
        'torch>=1.8.1+cu111'

    Raises:
        ValueError: When the requirement has more than two version criterion.

    Returns:
        str: Updated torch package with the right cuda suffix.

    """
    name = requirement.name

    for i, (operator, version) in enumerate(requirement.specs):
        updated_version = update_version_with_cuda_suffix(name, version)
        requirement.specs[i] = (operator, updated_version)

    # ``specs`` contains operators and versions as follows:
    # [('<=', '1.9.1+cu111'), ('>=', '1.8.1+cu111')]
    # These are to be concatenated again for the updated version.
    specs = [spec[0] + spec[1] for spec in requirement.specs]
    updated_requirement: str

    if specs:
        # This is the case when specs are e.g. ['<=1.9.1+cu111']
        if len(specs) == 1:
            updated_requirement = name + specs[0]
        # This is the case when specs are e.g., ['<=1.9.1+cu111', '>=1.8.1+cu111']
        elif len(specs) == 2:
            updated_requirement = name + specs[0] + ", " + specs[1]
        else:
            raise ValueError(
                f"Requirement version can be a single value or a range. \n"
                f"For example it could be torch>=1.8.1 or torch>=1.8.1, <=1.9.1\n"
                f"Got {specs} instead."
            )

    return updated_requirement


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
                    requirement = Requirement.parse(package)
                    if requirement.name in ("torch", "torchvision"):
                        package = update_torch_requirement(requirement)
                    requirements.append(package)

    return requirements


REQUIRED_PACKAGES = get_requirements(requirement_files=["base", "dev", "openvino"])
EXTRAS_REQUIRE = {
    "action": get_requirements(requirement_files="action"),
    "anomaly": get_requirements(requirement_files="anomaly"),
    "classification": get_requirements(requirement_files="classification"),
    "detection": get_requirements(requirement_files="detection"),
    "segmentation": get_requirements(requirement_files="segmentation"),
    "mpa": get_requirements(requirement_files=["classification", "detection", "segmentation", "action"]),
    "full": get_requirements(requirement_files=["anomaly", "classification", "detection", "segmentation", "action"]),
}
DEPENDENCY_LINKS = ["https://download.pytorch.org/whl/torch_stable.html"]


setup(
    name="otx",
    version=get_otx_version(),
    packages=find_packages(exclude=("tests",)),
    package_data={"": ["requirements.txt", "README.md", "LICENSE"]},  # Needed for exportable code
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,
    dependency_links=DEPENDENCY_LINKS,
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
