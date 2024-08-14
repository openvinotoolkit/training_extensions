# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX installation util functions."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess  # nosec B404
from importlib.metadata import requires
from importlib.util import find_spec
from pathlib import Path
from warnings import warn

import pkg_resources
from importlib_resources import files
from pkg_resources import Requirement

AVAILABLE_TORCH_VERSIONS = {
    "1.13.0": {"torchvision": "0.14.0", "cuda": ("11.6", "11.7")},
    "1.13.1": {"torchvision": "0.14.1", "cuda": ("11.6", "11.7")},
    "2.0.0": {"torchvision": "0.15.1", "cuda": ("11.7", "11.8")},
    "2.0.1": {"torchvision": "0.15.2", "cuda": ("11.7", "11.8")},
    "2.1.1": {"torchvision": "0.16.1", "cuda": ("11.8", "12.1")},
    "2.1.2": {"torchvision": "0.16.2", "cuda": ("11.8", "12.1")},
    "2.2.0": {"torchvision": "0.17.0", "cuda": ("11.8", "12.1")},
    "2.2.1": {"torchvision": "0.17.1", "cuda": ("11.8", "12.1")},
    "2.2.2": {"torchvision": "0.17.2", "cuda": ("11.8", "12.1")},
    "2.3.0": {"torchvision": "0.18.0", "cuda": ("11.8", "12.1")},
    "2.3.1": {"torchvision": "0.18.1", "cuda": ("11.8", "12.1")},
    "2.4.0": {"torchvision": "0.19.0", "cuda": ("11.8", "12.1", "12.4")},
}

MM_REQUIREMENTS = [
    "mmcv",
    "mmcv-full",
    "mmengine",
    "mmdet",
    "mmsegmentation",
    "mmpretrain",
    "mmdeploy",
]


def get_requirements(module: str = "otx") -> dict[str, list[Requirement]]:
    """Get requirements of module from importlib.metadata.

    This function returns list of required packages from importlib_metadata.

    Example:
        >>> get_requirements("otx")
        {
            "api": ["attrs>=21.2.0", ...],
            "anomaly": ["anomalib==0.5.1", ...],
            ...
        }

    Returns:
        dict[str, list[Requirement]]: List of required packages for each optional-extras.
    """
    requirement_list: list[str] | None = requires(module)
    extra_requirement: dict[str, list[Requirement]] = {}
    if requirement_list is None:
        return extra_requirement
    for requirement in requirement_list:
        extra = "api"
        requirement_extra: list[str] = requirement.replace(" ", "").split(";")
        if isinstance(requirement_extra, list) and len(requirement_extra) > 1:
            extra = requirement_extra[-1].split("==")[-1].strip("'\"")
        _requirement_name = requirement_extra[0]
        _requirement = Requirement.parse(_requirement_name)
        if extra in extra_requirement:
            extra_requirement[extra].append(_requirement)
        else:
            extra_requirement[extra] = [_requirement]
    return extra_requirement


def parse_requirements(
    requirements: list[Requirement],
) -> tuple[str, list[str], list[str]]:
    """Parse requirements and returns torch, mmcv and task requirements.

    Args:
        requirements (list[Requirement]): List of requirements.

    Raises:
        ValueError: If torch requirement is not found.

    Examples:
        >>> requirements = [
        ...     Requirement.parse("torch==1.13.0"),
        ...     Requirement.parse("mmcv-full==1.7.0"),
        ...     Requirement.parse("mmcls==0.12.0"),
        ...     Requirement.parse("onnx>=1.8.1"),
        ... ]
        >>> parse_requirements(requirements=requirements)
        (Requirement.parse("torch==1.13.0"),
        [Requirement.parse("mmcv-full==1.7.0"), Requirement.parse("mmcls==0.12.0")],
        Requirement.parse("onnx>=1.8.1"))

    Returns:
        tuple[str, list[str], list[str]]: Tuple of torch, mmcv and other requirements.
    """
    torch_requirement: str | None = None
    mm_requirements: list[str] = []
    other_requirements: list[str] = []

    for requirement in requirements:
        if requirement.unsafe_name == "torch":
            torch_requirement = str(requirement)
            if len(requirement.specs) > 1:
                warn(
                    "requirements.txt contains. Please remove other versions of torch from requirements.",
                    stacklevel=2,
                )

        elif requirement.unsafe_name in MM_REQUIREMENTS:
            mm_requirements.append(str(requirement))

        # Rest of the requirements are task requirements.
        # Other torch-related requirements such as `torchvision` are to be excluded.
        # This is because torch-related requirements are already handled in torch_requirement.
        else:
            # if not requirement.unsafe_name.startswith("torch"):
            other_requirements.append(str(requirement))

    if not torch_requirement:
        msg = "Could not find torch requirement. OTX depends on torch. Please add torch to your requirements."
        raise ValueError(
            msg,
        )

    # Get the unique list of the requirements.
    mm_requirements = list(set(mm_requirements))
    other_requirements = list(set(other_requirements))

    return torch_requirement, mm_requirements, other_requirements


def get_cuda_version() -> str | None:
    """Get CUDA version installed on the system.

    Examples:
        >>> # Assume that CUDA version is 11.2
        >>> get_cuda_version()
        "11.2"

        >>> # Assume that CUDA is not installed on the system
        >>> get_cuda_version()
        None

    Returns:
        str | None: CUDA version installed on the system.
    """
    # 1. Check CUDA_HOME Environment variable
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

    if Path(cuda_home).exists():
        # Check $CUDA_HOME/version.json file.
        version_file = Path(cuda_home) / "version.json"
        if version_file.is_file():
            with Path(version_file).open() as file:
                data = json.load(file)
                cuda_version = data.get("cuda", {}).get("version", None)
                if cuda_version is not None:
                    cuda_version_parts = cuda_version.split(".")
                    return ".".join(cuda_version_parts[:2])
    # 2. 'nvcc --version' check & without version.json case
    try:
        result = subprocess.run(args=["nvcc", "--version"], capture_output=True, text=True, check=False)
        output = result.stdout

        cuda_version_pattern = r"cuda_(\d+\.\d+)"
        cuda_version_match = re.search(cuda_version_pattern, output)

        if cuda_version_match is not None:
            return cuda_version_match.group(1)
    except Exception:
        msg = "Could not find cuda-version. Instead, the CPU version of torch will be installed."
        warn(msg, stacklevel=2)
    return None


def update_cuda_version_with_available_torch_cuda_build(cuda_version: str, torch_version: str) -> str:
    """Update the installed CUDA version with the highest supported CUDA version by PyTorch.

    Args:
        cuda_version (str): The installed CUDA version.
        torch_version (str): The PyTorch version.

    Raises:
        Warning: If the installed CUDA version is not supported by PyTorch.

    Examples:
        >>> update_cuda_version_with_available_torch_cuda_builds("11.1", "1.13.0")
        "11.6"

        >>> update_cuda_version_with_available_torch_cuda_builds("11.7", "1.13.0")
        "11.7"

        >>> update_cuda_version_with_available_torch_cuda_builds("11.8", "1.13.0")
        "11.7"

        >>> update_cuda_version_with_available_torch_cuda_builds("12.1", "2.0.1")
        "11.8"

    Returns:
        str: The updated CUDA version.
    """
    max_supported_cuda = max(AVAILABLE_TORCH_VERSIONS[torch_version]["cuda"])
    min_supported_cuda = min(AVAILABLE_TORCH_VERSIONS[torch_version]["cuda"])
    bounded_cuda_version = max(min(cuda_version, max_supported_cuda), min_supported_cuda)

    if cuda_version != bounded_cuda_version:
        warn(
            f"Installed CUDA version is v{cuda_version}. \n"
            f"v{min_supported_cuda} <= Supported CUDA version <= v{max_supported_cuda}.\n"
            f"This script will use CUDA v{bounded_cuda_version}.\n"
            f"However, this may not be safe, and you are advised to install the correct version of CUDA.\n"
            f"For more details, refer to https://pytorch.org/get-started/locally/",
            stacklevel=2,
        )
        cuda_version = bounded_cuda_version

    return cuda_version


def get_cuda_suffix(cuda_version: str) -> str:
    """Get CUDA suffix for PyTorch or mmX versions.

    Args:
        cuda_version (str): CUDA version installed on the system.

    Note:
        The CUDA version of PyTorch is not always the same as the CUDA version
            that is installed on the system. For example, the latest PyTorch
            version (1.10.0) supports CUDA 11.3, but the latest CUDA version
            that is available for download is 11.2. Therefore, we need to use
            the latest available CUDA version for PyTorch instead of the CUDA
            version that is installed on the system. Therefore, this function
            shoudl be regularly updated to reflect the latest available CUDA.

    Examples:
        >>> get_cuda_suffix(cuda_version="11.2")
        "cu112"

        >>> get_cuda_suffix(cuda_version="11.8")
        "cu118"

    Returns:
        str: CUDA suffix for PyTorch or mmX version.
    """
    return f"cu{cuda_version.replace('.', '')}"


def get_hardware_suffix(with_available_torch_build: bool = False, torch_version: str | None = None) -> str:
    """Get hardware suffix for PyTorch or mmX versions.

    Args:
        with_available_torch_build (bool): Whether to use the latest available
            PyTorch build or not. If True, the latest available PyTorch build
            will be used. If False, the installed PyTorch build will be used.
            Defaults to False.
        torch_version (str | None): PyTorch version. This is only used when the
            ``with_available_torch_build`` is True.

    Examples:
        >>> # Assume that CUDA version is 11.2
        >>> get_hardware_suffix()
        "cu112"

        >>> # Assume that CUDA is not installed on the system
        >>> get_hardware_suffix()
        "cpu"

        Assume that that installed CUDA version is 12.1.
        However, the latest available CUDA version for PyTorch v2.0 is 11.8.
        Therefore, we use 11.8 instead of 12.1. This is because PyTorch does not
        support CUDA 12.1 yet. In this case, we could correct the CUDA version
        by setting `with_available_torch_build` to True.

        >>> cuda_version = get_cuda_version()
        "12.1"
        >>> get_hardware_suffix(with_available_torch_build=True, torch_version="2.0.1")
        "cu118"

    Returns:
        str: Hardware suffix for PyTorch or mmX version.
    """
    cuda_version = get_cuda_version()
    if cuda_version:
        if with_available_torch_build:
            if torch_version is None:
                msg = "``torch_version`` must be provided when with_available_torch_build is True."
                raise ValueError(msg)
            cuda_version = update_cuda_version_with_available_torch_cuda_build(cuda_version, torch_version)
        hardware_suffix = get_cuda_suffix(cuda_version)
    else:
        hardware_suffix = "cpu"

    return hardware_suffix


def add_hardware_suffix_to_torch(
    requirement: Requirement,
    hardware_suffix: str | None = None,
    with_available_torch_build: bool = False,
) -> str:
    """Add hardware suffix to the torch requirement.

    Args:
        requirement (Requirement): Requirement object comprising requirement
            details.
        hardware_suffix (str | None): Hardware suffix. If None, it will be set
            to the correct hardware suffix. Defaults to None.
        with_available_torch_build (bool): To check whether the installed
            CUDA version is supported by the latest available PyTorch build.
            Defaults to False.

    Examples:
        >>> from pkg_resources import Requirement
        >>> req = "torch>=1.13.0, <=2.0.1"
        >>> requirement = Requirement.parse(req)
        >>> requirement.name, requirement.specs
        ('torch', [('>=', '1.13.0'), ('<=', '2.0.1')])

        >>> add_hardware_suffix_to_torch(requirement)
        'torch>=1.13.0+cu121, <=2.0.1+cu121'

        ``with_available_torch_build=True`` will use the latest available PyTorch build.
        >>> req = "torch==2.0.1"
        >>> requirement = Requirement.parse(req)
        >>> add_hardware_suffix_to_torch(requirement, with_available_torch_build=True)
        'torch==2.0.1+cu118'

        It is possible to pass the ``hardware_suffix`` manually.
        >>> req = "torch==2.0.1"
        >>> requirement = Requirement.parse(req)
        >>> add_hardware_suffix_to_torch(requirement, hardware_suffix="cu121")
        'torch==2.0.1+cu111'

    Raises:
        ValueError: When the requirement has more than two version criterion.

    Returns:
        str: Updated torch package with the right cuda suffix.
    """
    name = requirement.unsafe_name
    updated_specs: list[str] = []

    for operator, version in requirement.specs:
        hardware_suffix = hardware_suffix or get_hardware_suffix(with_available_torch_build, version)
        updated_version = version + f"+{hardware_suffix}" if version not in ("2.1.0", "2.1.1") else version

        # ``specs`` contains operators and versions as follows:
        # These are to be concatenated again for the updated version.
        updated_specs.append(operator + updated_version)

    updated_requirement: str = ""

    if updated_specs:
        # This is the case when specs are e.g. ['<=1.9.1+cu111']
        if len(updated_specs) == 1:
            updated_requirement = name + updated_specs[0]
        # This is the case when specs are e.g., ['<=1.9.1+cu111', '>=1.8.1+cu111']
        elif len(updated_specs) == 2:
            updated_requirement = name + updated_specs[0] + ", " + updated_specs[1]
        else:
            msg = (
                "Requirement version can be a single value or a range. \n"
                "For example it could be torch>=1.8.1 "
                "or torch>=1.8.1, <=1.9.1\n"
                f"Got {updated_specs} instead."
            )
            raise ValueError(msg)
    return updated_requirement


def get_torch_install_args(requirement: str | Requirement) -> list[str]:
    """Get the install arguments for Torch requirement.

    This function will return the install arguments for the Torch requirement
    and its corresponding torchvision requirement.

    Args:
        requirement (str | Requirement): The torch requirement.

    Raises:
        RuntimeError: If the OS is not supported.

    Example:
        >>> from pkg_resources import Requirement
        >>> requriment = "torch>=1.13.0"
        >>> get_torch_install_args(requirement)
        ['--extra-index-url', 'https://download.pytorch.org/whl/cpu',
        'torch==1.13.0+cpu', 'torchvision==0.14.0+cpu']

    Returns:
        list[str]: The install arguments.
    """
    if isinstance(requirement, str):
        requirement = Requirement.parse(requirement)

    # NOTE: This does not take into account if the requirement has multiple versions
    #   such as torch<2.0.1,>=1.13.0
    if len(requirement.specs) < 1:
        return [str(requirement)]
    operator, version = requirement.specs[0]
    install_args: list[str] = []

    if platform.system() in ("Linux", "Windows"):
        # Get the hardware suffix (eg., +cpu, +cu116 and +cu118 etc.)
        hardware_suffix = get_hardware_suffix(with_available_torch_build=True, torch_version=version)

        # Create the PyTorch Index URL to download the correct wheel.
        index_url = f"https://download.pytorch.org/whl/{hardware_suffix}"

        # Create the PyTorch version depending on the CUDA version. For example,
        # If CUDA version is 11.2, then the PyTorch version is 1.8.0+cu112.
        # If CUDA version is None, then the PyTorch version is 1.8.0+cpu.
        torch_version = add_hardware_suffix_to_torch(requirement, hardware_suffix, with_available_torch_build=True)

        # Get the torchvision version depending on the torch version.
        torchvision_version = AVAILABLE_TORCH_VERSIONS[version]["torchvision"]
        torchvision_requirement = f"torchvision{operator}{torchvision_version}"
        if torchvision_version not in ("0.16.0", "0.16.1"):
            torchvision_requirement += f"+{hardware_suffix}"

        # Return the install arguments.
        install_args += [
            "--extra-index-url",
            # "--index-url",
            index_url,
            torch_version,
            torchvision_requirement,
        ]
    elif platform.system() in ("macos", "Darwin"):
        torch_version = str(requirement)
        install_args += [torch_version]
    else:
        msg = f"Unsupported OS: {platform.system()}"
        raise RuntimeError(msg)

    return install_args


def get_mmcv_install_args(torch_requirement: str | Requirement, mmcv_requirements: list[str]) -> list[str]:
    """Get the install arguments for MMCV.

    Args:
        torch_requirement (str | Requirement): Torch requirement.
        mmcv_requirements (list[str]): MMCV requirements.

    Raises:
        NotImplementedError: Not implemented for MacOS.
        RuntimeError: If the OS is not supported.

    Returns:
        list[str]: List of mmcv install arguments.
    """
    if isinstance(torch_requirement, str):
        torch_requirement = Requirement.parse(torch_requirement)

    if platform.system() in ("Linux", "Windows", "Darwin", "macos"):
        # Get the hardware suffix (eg., +cpu, +cu116 and +cu118 etc.)
        _, version = torch_requirement.specs[0]
        hardware_suffix = get_hardware_suffix(with_available_torch_build=True, torch_version=version)

        # MMCV builds are only available for major.minor.0 torch versions.
        major, minor, _ = version.split(".")
        mmcv_torch_version = f"{major}.{minor}.0"
        mmcv_index_url = (
            f"https://download.openmmlab.com/mmcv/dist/{hardware_suffix}/torch{mmcv_torch_version}/index.html"
        )

        # Return the install arguments.
        return ["--find-links", mmcv_index_url, *mmcv_requirements]

    msg = f"Unsupported OS: {platform.system()}"
    raise RuntimeError(msg)


def mim_installation(requirements: list[str]) -> int:
    """Installing libraries with mim api.

    Args:
        requirements (list[str]): List of MMCV-related libraries.

    Raises:
        ModuleNotFoundError: Raise an error if mim import is not possible.
    """
    if not find_spec("mim"):
        msg = "The mmX library installation requires mim. mim is not currently installed."
        raise ModuleNotFoundError(msg)
    from mim import install

    return install(requirements)


def get_module_version(module_name: str) -> str | None:
    """Return the version of the specified Python module.

    Args:
        module_name (str): The name of the module to get the version of.

    Returns:
        str | None: The version of the module, or None if the module is not installed.
    """
    try:
        module_version = pkg_resources.get_distribution(module_name).version
    except pkg_resources.DistributionNotFound:
        module_version = None

    return module_version


def patch_mmaction2() -> None:
    """Patch MMAction2==1.2.0 with the custom code.

    The patch is at `src/otx/cli/patches/mmaction2.patch`.
    The reason why we need is that `__init__.py` is missing in
    https://github.com/open-mmlab/mmaction2/tree/v1.2.0/mmaction/models/localizers/drn
    """
    dir_patches: Path = files("otx") / "cli" / "patches"
    file_mmaction2_patch = dir_patches / "mmaction2.patch"

    if not file_mmaction2_patch.exists():
        msg = f"Cannot find `mmaction2.patch` file from {dir_patches}"
        warn(msg, stacklevel=1)
        return

    if (spec := find_spec("mmaction")) is None:
        msg = "Cannot find mmaction spec"
        warn(msg, stacklevel=1)
        return

    if (spec_origin := spec.origin) is None:
        msg = "Cannot find mmaction spec origin"
        warn(msg, stacklevel=1)
        return

    dir_mmaction_parent = Path(spec_origin).parent.parent

    proc = subprocess.Popen(
        args=["patch", "-p0", "--forward"],
        cwd=dir_mmaction_parent,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate(input=file_mmaction2_patch.read_bytes(), timeout=1)

    if proc.returncode > 1:
        msg = f"Cannot patch. Error code: {proc.returncode}, stdout: {stdout.decode()} and stderr: {stderr.decode()}"
        warn(msg, stacklevel=1)
        return
    if proc.returncode == 1:
        warn("MMAction2 is already patched. Skip patching it.", stacklevel=1)
