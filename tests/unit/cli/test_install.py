# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from _pytest.monkeypatch import MonkeyPatch
from jsonargparse import ArgumentParser
from otx.cli.install import add_install_parser, otx_install
from pkg_resources import Requirement
from pytest_mock.plugin import MockerFixture


class TestInstall:
    @pytest.fixture(autouse=True)
    def setup(self, mocker: MockerFixture) -> None:
        requirements_dict = {
            "base": [Requirement.parse("torch==2.0.0"), Requirement.parse("pytorchcv")],
            "openvino": [Requirement.parse("openvino")],
            "mmlab": [Requirement.parse("mmpretrain")],
            "api": [Requirement.parse("test1")],
            "xpu": [Requirement.parse("torch==2.0.0"), Requirement.parse("ipex")],
        }
        mocker.patch("otx.cli.install.get_requirements", return_value=requirements_dict)

    def test_add_install_parser(self) -> None:
        parser = ArgumentParser()
        parser_subcommands = parser.add_subcommands()
        add_install_parser(parser_subcommands)
        assert parser_subcommands.choices.get("install") is not None
        install_parser = parser_subcommands.choices.get("install")
        argument_list = [action.dest for action in install_parser._actions]
        expected_argument = ["help", "option", "verbose", "do_not_install_torch", "user"]
        assert argument_list == expected_argument

    def test_install_extra(self, mocker: MockerFixture) -> None:
        mock_create_command = mocker.patch("pip._internal.commands.create_command")
        mock_create_command.return_value.main.return_value = 0
        status_code = otx_install(option="dev")
        assert status_code == mock_create_command.return_value.main.return_value

        argument_call_list = []
        for call_args in mock_create_command.return_value.main.call_args_list:
            for arg in call_args.args:
                argument_call_list += arg

        assert "pytorchcv" in argument_call_list
        assert "openvino" not in argument_call_list

    def test_install_full(self, mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
        mock_create_command = mocker.patch("pip._internal.commands.create_command")
        mock_create_command.return_value.main.return_value = 0
        mock_mim_installation = mocker.patch("otx.cli.install.mim_installation")
        mock_mim_installation.return_value = 0

        status_code = otx_install("full")
        assert status_code == mock_create_command.return_value.main.return_value
        mock_create_command.assert_called_with("install")

        argument_call_list = []
        for call_args in mock_create_command.return_value.main.call_args_list:
            for arg in call_args.args:
                argument_call_list += arg

        assert "openvino" in argument_call_list
        assert "pytorchcv" in argument_call_list
        assert "mmpretrain" not in argument_call_list
        assert "ipex" not in argument_call_list
        mm_argument_call_list = mock_mim_installation.call_args_list[-1][0][-1]
        assert "mmpretrain" in mm_argument_call_list
