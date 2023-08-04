# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Optional

import pytest
from otx.v2.cli.extensions.doctor import add_doctor_parser, doctor, main, prepare_parser
from otx.v2.cli.utils.arg_parser import OTXArgumentParser
from pytest_mock.plugin import MockerFixture


def test_prepare_parser() -> None:
    parser = prepare_parser()
    assert parser is not None
    argument_list = [action.dest for action in parser._actions]
    expected_argument = ["help", "task", "verbose"]
    assert argument_list == expected_argument


def test_add_doctor_parser() -> None:
    parser = OTXArgumentParser()
    parser_subcommands = parser.add_subcommands()
    add_doctor_parser(parser_subcommands)
    assert parser_subcommands.choices.get("doctor") is not None


@pytest.mark.parametrize("torch_version", [None, 1.0])
@pytest.mark.parametrize("cuda_available", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
def test_doctor(mocker: MockerFixture, verbose: bool, cuda_available: bool, torch_version: Optional[float]) -> None:
    mock_get_environment_table = mocker.patch("otx.v2.cli.extensions.doctor.get_environment_table", return_value="environment_table")
    mock_check_torch_cuda = mocker.patch("otx.v2.cli.extensions.doctor.check_torch_cuda", return_value=(torch_version, cuda_available))
    mock_get_task_status = mocker.patch("otx.v2.cli.extensions.doctor.get_task_status", return_value={"task1": {"AVAILABLE": True, "EXCEPTIONS": []}})
    mock_console = mocker.patch("otx.v2.cli.extensions.doctor.Console", return_value=mocker.MagicMock())

    doctor(task="task1", verbose=verbose)
    mock_get_environment_table.assert_called_once_with(task="task1", verbose=verbose)
    mock_check_torch_cuda.assert_called_once_with()
    mock_get_task_status.assert_called_once_with(task="task1")
    mock_console.assert_called_once()

    mock_get_task_status = mocker.patch("otx.v2.cli.extensions.doctor.get_task_status", return_value={"task1": {"AVAILABLE": True, "EXCEPTIONS": []}, "task2": {"AVAILABLE": False, "EXCEPTIONS": [ValueError("test")]}})
    doctor(task="task2", verbose=verbose)
    mock_get_task_status.assert_called_once_with(task="task2")
    mock_console.assert_called()


def test_doctor_main(mocker: MockerFixture) -> None:
    mock_get_environment_table = mocker.patch("otx.v2.cli.extensions.doctor.get_environment_table", return_value="environment_table")
    mock_check_torch_cuda = mocker.patch("otx.v2.cli.extensions.doctor.check_torch_cuda", return_value=(None, False))
    mock_get_task_status = mocker.patch("otx.v2.cli.extensions.doctor.get_task_status", return_value={"task1": {"AVAILABLE": True, "EXCEPTIONS": []}})
    mock_console = mocker.patch("otx.v2.cli.extensions.doctor.Console", return_value=mocker.MagicMock())
    argv = ["otx", "doctor"]
    mocker.patch.object(sys, "argv", argv)
    main()
    mock_get_environment_table.assert_called_once()
    mock_check_torch_cuda.assert_called_once()
    mock_get_task_status.assert_called_once()
    mock_console.assert_called_once()
