# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import pkg_resources
import pytest
from otx.cli.utils.installation import (
    add_hardware_suffix_to_torch,
    get_cuda_suffix,
    get_cuda_version,
    get_hardware_suffix,
    get_mmcv_install_args,
    get_module_version,
    get_requirements,
    get_torch_install_args,
    mim_installation,
    parse_requirements,
    update_cuda_version_with_available_torch_cuda_build,
)
from pkg_resources import Requirement
from pytest_mock import MockerFixture

SKIP_MMLAB_TEST = False
try:
    import mim  # noqa: F401
except ImportError:
    SKIP_MMLAB_TEST = True


@pytest.fixture()
def requirements_file() -> Path:
    """Create a temporary requirements file with some example requirements."""
    requirements = ["numpy==1.19.5", "opencv-python-headless>=4.5.1.48"]
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("\n".join(requirements))
        return Path(f.name)


def test_get_requirements(mocker: MockerFixture) -> None:
    """Test that get_requirements returns the expected dictionary of requirements."""
    requirements = get_requirements("otx")
    assert isinstance(requirements, dict)
    assert len(requirements) > 0
    for reqs in requirements.values():
        assert isinstance(reqs, list)
        for req in reqs:
            assert isinstance(req, Requirement)
    mocker.patch("otx.cli.utils.installation.requires", return_value=None)
    assert get_requirements() == {}


def test_parse_requirements() -> None:
    """Test that parse_requirements returns the expected tuple of requirements."""
    requirements = [
        Requirement.parse("torch==2.0.0"),
        Requirement.parse("mmengine==2.0.0"),
        Requirement.parse("mmpretrain==0.12.0"),
        Requirement.parse("onnx>=1.8.1"),
    ]
    torch_req, mm_reqs, other_reqs = parse_requirements(requirements)
    assert isinstance(torch_req, str)
    assert isinstance(mm_reqs, list)
    assert isinstance(other_reqs, list)
    assert torch_req == "torch==2.0.0"
    assert "mmengine==2.0.0" in mm_reqs
    assert "mmpretrain==0.12.0" in mm_reqs
    assert other_reqs == ["onnx>=1.8.1"]

    requirements = [
        Requirement.parse("torch<=2.0.1, >=1.8.1"),
    ]
    torch_req, mm_reqs, other_reqs = parse_requirements(requirements)
    assert torch_req == "torch<=2.0.1,>=1.8.1"
    assert mm_reqs == []
    assert other_reqs == []

    requirements = [
        Requirement.parse("onnx>=1.8.1"),
    ]
    with pytest.raises(ValueError, match="Could not find torch requirement."):
        parse_requirements(requirements)


def test_get_cuda_version_with_version_file(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test that get_cuda_version returns the expected CUDA version when version file exists."""
    tmp_path = tmp_path / "cuda"
    tmp_path.mkdir()
    mocker.patch.dict(os.environ, {"CUDA_HOME": str(tmp_path)})
    version_file = tmp_path / "version.json"
    version_file.write_text('{"cuda": {"version": "11.2.0"}}')
    assert get_cuda_version() == "11.2"


def test_get_cuda_version_with_nvcc(mocker: MockerFixture) -> None:
    """Test that get_cuda_version returns the expected CUDA version when nvcc is available."""
    mock_run = mocker.patch("otx.cli.utils.installation.Path.exists", return_value=False)
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "Build cuda_11.2.r11.2/compiler.00000_0"
    assert get_cuda_version() == "11.2"

    mock_run = mocker.patch("subprocess.run")
    mock_run.side_effect = FileNotFoundError
    assert get_cuda_version() is None


def test_update_cuda_version_with_available_torch_cuda_build() -> None:
    """Test that update_cuda_version_with_available_torch_cuda_build returns the expected CUDA version."""
    assert update_cuda_version_with_available_torch_cuda_build("11.1", "1.13.0") == "11.6"
    assert update_cuda_version_with_available_torch_cuda_build("11.7", "1.13.0") == "11.7"
    assert update_cuda_version_with_available_torch_cuda_build("11.8", "1.13.0") == "11.7"
    assert update_cuda_version_with_available_torch_cuda_build("12.1", "2.0.1") == "11.8"


def test_get_cuda_suffix() -> None:
    assert get_cuda_suffix(cuda_version="11.2") == "cu112"
    assert get_cuda_suffix(cuda_version="11.8") == "cu118"


def test_get_hardware_suffix(mocker: MockerFixture) -> None:
    mocker.patch("otx.cli.utils.installation.get_cuda_version", return_value="11.2")
    assert get_hardware_suffix() == "cu112"

    mocker.patch("otx.cli.utils.installation.get_cuda_version", return_value="12.1")
    assert get_hardware_suffix(with_available_torch_build=True, torch_version="2.0.1") == "cu118"

    with pytest.raises(ValueError, match="``torch_version`` must be provided"):
        get_hardware_suffix(with_available_torch_build=True)

    mocker.patch("otx.cli.utils.installation.get_cuda_version", return_value=None)
    assert get_hardware_suffix() == "cpu"


def test_add_hardware_suffix_to_torch(mocker: MockerFixture) -> None:
    """Test that add_hardware_suffix_to_torch returns the expected updated requirement."""
    mocker.patch("otx.cli.utils.installation.get_hardware_suffix", return_value="cu121")
    requirement = Requirement.parse("torch>=1.13.0, <=2.0.1")
    updated_requirement = add_hardware_suffix_to_torch(requirement)
    assert "torch" in updated_requirement
    assert ">=1.13.0+cu121" in updated_requirement
    assert "<=2.0.1+cu121" in updated_requirement

    requirement = Requirement.parse("torch==2.0.1")
    mocker.patch("otx.cli.utils.installation.get_hardware_suffix", return_value="cu118")
    updated_requirement = add_hardware_suffix_to_torch(requirement, with_available_torch_build=True)
    assert updated_requirement == "torch==2.0.1+cu118"

    requirement = Requirement.parse("torch==2.0.1")
    updated_requirement = add_hardware_suffix_to_torch(requirement, hardware_suffix="cu111")
    assert updated_requirement == "torch==2.0.1+cu111"

    requirement = Requirement.parse("torch>=1.13.0, <=2.0.1, !=1.14.0")
    with pytest.raises(ValueError, match="Requirement version can be a single value or a range."):
        add_hardware_suffix_to_torch(requirement)


def test_get_torch_install_args(mocker: MockerFixture) -> None:
    """Test that get_torch_install_args returns the expected install arguments."""
    requirement = Requirement.parse("torch>=1.13.0")
    mocker.patch("otx.cli.utils.installation.platform.system", return_value="Linux")
    mocker.patch("otx.cli.utils.installation.get_hardware_suffix", return_value="cpu")
    install_args = get_torch_install_args(requirement)
    expected_args = [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "torch>=1.13.0+cpu",
        "torchvision>=0.14.0+cpu",
    ]
    for arg in expected_args:
        assert arg in install_args

    requirement = Requirement.parse("torch>=1.13.0,<=2.0.1")
    mocker.patch("otx.cli.utils.installation.get_hardware_suffix", return_value="cu111")
    install_args = get_torch_install_args(requirement)
    expected_args = [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu111",
    ]
    for arg in expected_args:
        assert arg in install_args

    requirement = Requirement.parse("torch==2.0.1")
    expected_args = [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu111",
        "torch==2.0.1+cu111",
        "torchvision==0.15.2+cu111",
    ]
    install_args = get_torch_install_args(requirement)
    for arg in expected_args:
        assert arg in install_args

    install_args = get_torch_install_args("torch")
    assert install_args == ["torch"]

    mocker.patch("otx.cli.utils.installation.platform.system", return_value="Darwin")
    requirement = Requirement.parse("torch==2.0.1")
    install_args = get_torch_install_args(requirement)
    assert install_args == ["torch==2.0.1"]

    mocker.patch("otx.cli.utils.installation.platform.system", return_value="Unknown")
    with pytest.raises(RuntimeError, match="Unsupported OS: Unknown"):
        get_torch_install_args(requirement)


def test_get_mmcv_install_args(mocker: MockerFixture) -> None:
    """Test that get_mmcv_install_args returns the expected install arguments."""
    torch_requirement = Requirement.parse("torch>=1.13.0")
    mocker.patch("otx.cli.utils.installation.platform.system", return_value="Linux")
    mocker.patch("otx.cli.utils.installation.get_hardware_suffix", return_value="cpu")
    install_args = get_mmcv_install_args(torch_requirement=torch_requirement, mmcv_requirements=["mmengine==2.0.0"])
    expected_args = [
        "--find-links",
        "https://download.openmmlab.com/mmcv/dist/cpu/torch1.13.0/index.html",
        "mmengine==2.0.0",
    ]
    assert install_args == expected_args

    install_args = get_mmcv_install_args(torch_requirement="torch==2.0.1", mmcv_requirements=["mmengine==2.0.0"])
    expected_args = [
        "--find-links",
        "https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html",
        "mmengine==2.0.0",
    ]
    assert install_args == expected_args

    mocker.patch("otx.cli.utils.installation.platform.system", return_value="Unknown")
    with pytest.raises(RuntimeError, match="Unsupported OS: Unknown"):
        get_mmcv_install_args(torch_requirement=torch_requirement, mmcv_requirements=["mmengine==2.0.0"])


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
def test_mim_installation(mocker: MockerFixture) -> None:
    mocker.patch("otx.cli.utils.installation.find_spec", return_value=True)
    # https://github.com/Madoshakalaka/pipenv-setup/issues/101
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    mock_mim_install = mocker.patch("mim.install")
    mim_installation(["mmengine==2.0.0"])
    mock_mim_install.assert_called_once_with(["mmengine==2.0.0"])

    mocker.patch("otx.cli.utils.installation.find_spec", return_value=False)
    with pytest.raises(ModuleNotFoundError, match="The mmX library installation requires mim."):
        mim_installation(["mmengine==2.0.0"])


def test_get_module_version(mocker: MockerFixture) -> None:
    mock_get_distribution = mocker.patch("otx.cli.utils.installation.pkg_resources.get_distribution")
    mock_get_distribution.return_value.version = "1.0.0"
    version = get_module_version("test")
    assert version == "1.0.0"
    mock_get_distribution.assert_called_once_with("test")

    mock_get_distribution.side_effect = pkg_resources.DistributionNotFound
    version = get_module_version("test")
    assert version is None
