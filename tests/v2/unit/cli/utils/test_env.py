# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from _pytest.monkeypatch import MonkeyPatch
from otx.v2.cli.utils.env import (
    check_torch_cuda,
    get_adapters_status,
    get_environment_table,
    get_task_status,
)
from pkg_resources import Requirement
from pytest_mock import MockerFixture


class TestEnvironmentUtils:
    @pytest.fixture(autouse=True)
    def setup(self, mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
        mock_module = mocker.MagicMock()
        mock_module.AVAILABLE = True
        mock_module.VERSION = "1.0.0"
        mock_module.DEBUG = None
        mock_invalid_module = mocker.MagicMock()
        mock_invalid_module.AVAILABLE = False
        mock_invalid_module.VERSION = None
        mock_invalid_module.DEBUG = ModuleNotFoundError
        mocker.patch.dict(
            "sys.modules",
            {
                "otx.v2.adapters.adapter1": mock_module,
                "otx.v2.adapters.adapter2": mock_module,
                "otx.v2.adapters.adapter3": mock_invalid_module,
            },
        )
        monkeypatch.setattr("otx.v2.cli.utils.env.ADAPTERS", ["adapter1", "adapter2", "adapter3"])
        adapters_per_task = {
            "anomaly": ["adapter1", "adapter3"],
            "classification": [
                "adapter1",
                "adapter2",
            ],
        }
        monkeypatch.setattr("otx.v2.cli.utils.env.REQUIRED_ADAPTERS_PER_TASK", adapters_per_task)

    def test_get_adapters_status(self) -> None:
        expected_output = {
            "otx.v2.adapters.adapter1": {"AVAILABLE": True, "VERSION": "1.0.0", "DEBUG": None},
            "otx.v2.adapters.adapter2": {"AVAILABLE": True, "VERSION": "1.0.0", "DEBUG": None},
            "otx.v2.adapters.adapter3": {"AVAILABLE": False, "VERSION": None, "DEBUG": ModuleNotFoundError},
        }
        assert get_adapters_status() == expected_output


    def test_get_environment_table(self, mocker: MockerFixture) -> None:
        requirements_per_task = {
            "api": [Requirement.parse('adapter1>=1.0.0')],
            "classification": [Requirement.parse('adapter2>=1.0.0')],
            "anomaly": [Requirement.parse('adapter3>=1.0.0')],
        }

        mock_get_requirements = mocker.patch(
            "otx.v2.cli.utils.env.get_requirements", return_value=requirements_per_task,
        )

        table = get_environment_table()
        assert "classification" in table
        assert "anomaly" in table
        assert "adapter1" in table
        assert "adapter2" in table
        assert "adapter3" in table
        assert "1.0.0" in table
        mock_get_requirements.assert_called_once_with()

        table = get_environment_table(task="classification")
        assert "classification" in table
        assert "adapter1" in table
        assert "adapter2" in table
        assert "1.0.0" in table
        assert "anomaly" not in table

        table = get_environment_table(task="anomaly")
        assert "anomaly" in table
        assert "adapter1" in table
        assert "adapter3" in table
        assert "classification" not in table

    def test_get_environment_table_with_verbose(self, mocker: MockerFixture) -> None:
        requirements_per_task = {
            "api": [Requirement.parse('adapter1>=1.0.0')],
            "classification": [Requirement.parse('adapter2>=1.0.0')],
        }

        mock_get_requirements = mocker.patch(
            "otx.v2.cli.utils.env.get_requirements", return_value=requirements_per_task,
        )
        mock_get_module_version = mocker.patch(
            "otx.v2.cli.utils.env.get_module_version", return_value="2.0.0",
        )
        table = get_environment_table(task="classification", verbose=True)
        assert "classification" in table
        assert "adapter1" in table
        assert "adapter2" in table
        assert "2.0.0" in table
        mock_get_module_version.assert_has_calls(
            [mocker.call("adapter1"), mocker.call("adapter2")], any_order=True,
        )
        mock_get_requirements.assert_called_once_with()


    def test_get_task_status(self, monkeypatch: MonkeyPatch) -> None:
        expected_output = {
            "anomaly": {"AVAILABLE": False, "EXCEPTIONS": [ModuleNotFoundError]},
            "classification": {"AVAILABLE": True, "EXCEPTIONS": []},
        }
        assert get_task_status() == expected_output

        expected_output = {
            "anomaly": {"AVAILABLE": False, "EXCEPTIONS": [ModuleNotFoundError]},
        }
        assert get_task_status(task="anomaly") == expected_output

        adapters_per_task = {
            "anomaly": ["adapter1", "adapter4"],
        }
        monkeypatch.setattr("otx.v2.cli.utils.env.REQUIRED_ADAPTERS_PER_TASK", adapters_per_task)
        result = get_task_status(task="anomaly")
        assert "anomaly" in result
        assert result["anomaly"]["AVAILABLE"] is False
        assert len(result["anomaly"]["EXCEPTIONS"]) >= 1


    def test_check_torch_cuda(self, mocker: MockerFixture) -> None:
        mock_find_spec = mocker.patch("importlib.util.find_spec", return_value=None)
        assert check_torch_cuda() == (None, False)
        mock_find_spec.assert_called_once_with("torch")

        mock_torch_module = mocker.MagicMock()
        mock_torch_module.cuda.is_available.return_value = True
        mock_torch_module.__version__ = "1.0.0"
        mock_find_spec.return_value = True
        mocker.patch.dict("sys.modules", {"torch": mock_torch_module})
        assert check_torch_cuda() == ("1.0.0", True)
        mock_find_spec.assert_called_with("torch")
        mock_torch_module.cuda.is_available.assert_called_once_with()
