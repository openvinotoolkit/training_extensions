# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from _pytest.monkeypatch import MonkeyPatch
from otx.v2.cli.extensions.install import add_install_parser, install, main, prepare_parser
from otx.v2.cli.utils.arg_parser import OTXArgumentParser
from pkg_resources import Requirement
from pytest_mock.plugin import MockerFixture


class TestInstall:
    @pytest.fixture(autouse=True)
    def setup(self, mocker: MockerFixture) -> None:
        requirements_dict = {
            "base": [Requirement.parse('torch==2.0.0'), Requirement.parse('pytorchcv')],
            "openvino": [Requirement.parse('openvino')],
            "classification": [Requirement.parse('mmpretrain')],
            "anomaly": [Requirement.parse('anomalib')],
            "api": [Requirement.parse('test1')],
        }
        mocker.patch("otx.v2.cli.extensions.install.get_requirements", return_value=requirements_dict)

    def test_prepare_parser(self) -> None:
        parser = prepare_parser()
        assert parser is not None
        argument_list = [action.dest for action in parser._actions]
        expected_argument = ["help", "task"]
        assert argument_list == expected_argument

    def test_add_install_parser(self) -> None:
        parser = OTXArgumentParser()
        parser_subcommands = parser.add_subcommands()
        add_install_parser(parser_subcommands)
        assert parser_subcommands.choices.get("install") is not None

    def test_install_invalid_task(self) -> None:
        with pytest.raises(ValueError, match="Supported tasks:"):
            install("invalid")

    def test_install_without_mm(self, mocker: MockerFixture) -> None:
        mock_create_command = mocker.patch("otx.v2.cli.extensions.install.create_command")
        status_code = install("anomaly")
        assert status_code == mock_create_command.return_value.main.return_value
        mock_create_command.assert_called_once_with("install")
        argument_call_list = mock_create_command.return_value.main.call_args_list[-1][0][-1]
        assert "openvino" in argument_call_list
        assert "pytorchcv" in argument_call_list
        assert "anomalib" in argument_call_list

    def test_install_extra(self, mocker: MockerFixture) -> None:
        mock_create_command = mocker.patch("otx.v2.cli.extensions.install.create_command")
        status_code = install("api")
        assert status_code == mock_create_command.return_value.main.return_value
        argument_call_list = mock_create_command.return_value.main.call_args_list[-1][0][-1]
        assert "pytorchcv" in argument_call_list
        assert "test1" in argument_call_list
        assert "openvino" not in argument_call_list
        assert "anomalib" not in argument_call_list

    def test_install_with_mm(self, mocker: MockerFixture) -> None:
        mock_create_command = mocker.patch("otx.v2.cli.extensions.install.create_command")
        mock_mim_installation = mocker.patch("otx.v2.cli.extensions.install.mim_installation")

        status_code = install("classification")
        assert status_code == mock_create_command.return_value.main.return_value
        mock_create_command.assert_called_once_with("install")
        argument_call_list = mock_create_command.return_value.main.call_args_list[-1][0][-1]
        assert "openvino" in argument_call_list
        assert "pytorchcv" in argument_call_list
        assert "openmim" in argument_call_list
        assert "anomalib" not in argument_call_list
        mm_argument_call_list = mock_mim_installation.call_args_list[-1][0][-1]
        assert "mmpretrain" in mm_argument_call_list

    def test_install_full(self, mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
        mock_create_command = mocker.patch("otx.v2.cli.extensions.install.create_command")
        mock_mim_installation = mocker.patch("otx.v2.cli.extensions.install.mim_installation")
        monkeypatch.setattr("otx.v2.cli.extensions.install.SUPPORTED_TASKS", ["classification", "anomaly"])

        status_code = install("full")
        assert status_code == mock_create_command.return_value.main.return_value
        mock_create_command.assert_called_once_with("install")
        argument_call_list = mock_create_command.return_value.main.call_args_list[-1][0][-1]
        assert "openvino" in argument_call_list
        assert "pytorchcv" in argument_call_list
        assert "anomalib" in argument_call_list
        assert "mmpretrain" not in argument_call_list
        mm_argument_call_list = mock_mim_installation.call_args_list[-1][0][-1]
        assert "mmpretrain" in mm_argument_call_list

    def test_install_main(self, mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
        mock_create_command = mocker.patch("otx.v2.cli.extensions.install.create_command")
        mock_mim_installation = mocker.patch("otx.v2.cli.extensions.install.mim_installation")
        monkeypatch.setattr("otx.v2.cli.extensions.install.SUPPORTED_TASKS", ["classification", "anomaly"])

        mocker.patch.object(sys, "argv", ["full"])
        main()
        argument_call_list = mock_create_command.return_value.main.call_args_list[-1][0][-1]
        assert "openvino" in argument_call_list
        assert "pytorchcv" in argument_call_list
        assert "anomalib" in argument_call_list
        assert "mmpretrain" not in argument_call_list
        mm_argument_call_list = mock_mim_installation.call_args_list[-1][0][-1]
        assert "mmpretrain" in mm_argument_call_list
