# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import pytest

from otx.cli.utils import parser as target_package
from otx.cli.utils.parser import (
    MemSizeAction,
    add_hyper_parameters_sub_parser,
    gen_param_help,
    gen_params_dict_from_args,
    get_parser_and_hprams_data,
    str2bool,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit

FAKE_HYPER_PARAMETERS = {
    "a": {
        "a": "fake",
        "b": {
            "default_value": "default_value",
            "type": "SELECTABLE",
            "affects_outcome_of": "TRAINING",
            "header": "header",
        },
        "visible_in_ui": True,
    },
    "b": {
        "a": {
            "default_value": "default_value",
            "type": "BOOLEAN",
            "affects_outcome_of": "TRAINING",
            "header": "header",
        },
        "visible_in_ui": True,
    },
    "c": {
        "a": {
            "a": {
                "default_value": "default_value",
                "type": "INTEGER",
                "affects_outcome_of": "TRAINING",
                "header": "header",
            },
            "visible_in_ui": True,
        },
        "visible_in_ui": True,
    },
    "d": {
        "a": {"default_value": "default_value", "type": "FLOAT", "affects_outcome_of": "TRAINING", "header": "header"},
        "visible_in_ui": True,
    },
}


@e2e_pytest_unit
def test_gen_param_help():
    param_help = gen_param_help(FAKE_HYPER_PARAMETERS)

    hp_type_map = {
        "a.b": str,
        "b.a": bool,
        "c.a.a": int,
        "d.a": float,
    }
    for key, val in hp_type_map.items():
        assert param_help[key]["default"] == "default_value"
        assert param_help[key]["affects_outcome_of"] == "TRAINING"
        assert "help" in param_help[key]
        assert param_help[key]["type"] == val


@pytest.fixture
def mock_args(mocker):
    mock_args = mocker.Mock()
    setattr(mock_args, "params.a.a", 1)
    setattr(mock_args, "params.a.b", 2.1)
    setattr(mock_args, "params.a.c", True)
    setattr(mock_args, "params.b", "fake")
    setattr(mock_args, "params.c", 10)
    setattr(mock_args, "params.d", None)

    return mock_args


@e2e_pytest_unit
def test_gen_params_dict_from_args(mock_args):
    param_dict = gen_params_dict_from_args(
        mock_args, ["params.a.a", "params.a.b", "params.a.c", "params.b", "params.c"]
    )

    assert param_dict["a"]["a"]["value"] == 1
    assert param_dict["a"]["b"]["value"] == 2.1
    assert param_dict["a"]["c"]["value"] is True
    assert param_dict["b"]["value"] == "fake"
    assert param_dict["c"]["value"] == 10
    assert "d" not in param_dict


@e2e_pytest_unit
def test_gen_params_dict_from_args_with_type_hint(mock_args):
    type_hint = {
        "a.a": {"type": str},
        "a.b": {"type": int},
        "a.c": {"type": bool},
        "b": {"type": str},
        "c": {"type": str},
    }

    param_dict = gen_params_dict_from_args(
        mock_args, ["params.a.a", "params.a.b", "params.a.c", "params.b", "params.c"], type_hint
    )

    assert param_dict["a"]["a"]["value"] == "1"
    assert param_dict["a"]["b"]["value"] == 2
    assert param_dict["a"]["c"]["value"] is True
    assert param_dict["b"]["value"] == "fake"
    assert param_dict["c"]["value"] == "10"


@e2e_pytest_unit
@pytest.mark.parametrize("val", [True, False])
def test_str2bool_with_bool_input(val):
    assert str2bool(val) is val


@e2e_pytest_unit
@pytest.mark.parametrize("val", ["true", "1"])
def test_str2bool_with_bool_string_true(val):
    assert str2bool(val) is True


@e2e_pytest_unit
@pytest.mark.parametrize("val", ["false", "0"])
def test_str2bool_with_bool_string_false(val):
    assert str2bool(val) is False


@e2e_pytest_unit
@pytest.mark.parametrize("val", [1, 1.2, "abc"])
def test_str2bool_with_bool_wrong_input(val):
    with pytest.raises(ArgumentTypeError):
        assert str2bool(val)


@e2e_pytest_unit
def test_add_hyper_parameters_sub_parser(mocker):
    # prepare
    mock_parser = mocker.MagicMock()
    mock_subparser = mocker.MagicMock()
    mock_parser.add_subparsers.return_value.add_parser.return_value = mock_subparser

    # run
    parser = add_hyper_parameters_sub_parser(mock_parser, FAKE_HYPER_PARAMETERS, return_sub_parser=True)

    # check
    hp_type_map = {
        "a.b": str,
        "b.a": bool,
        "c.a.a": int,
        "d.a": float,
    }
    assert parser is not None
    add_args_call_args = mock_subparser.add_argument.call_args_list
    for i, key in enumerate(hp_type_map):
        assert add_args_call_args[i][0][0] == f"--{key}"
        assert add_args_call_args[i][1]["default"] == "default_value"
        assert "help" in add_args_call_args[i][1]
        assert add_args_call_args[i][1]["dest"] == f"params.{key}"

        hp_type = hp_type_map[key]
        if hp_type is not bool:
            assert add_args_call_args[i][1]["type"] == hp_type
        else:
            assert add_args_call_args[i][1]["type"] == str2bool


@e2e_pytest_unit
def test_get_parser_and_hprams_data_with_fake_template(mocker, tmp_dir):
    # prepare
    tmp_dir = Path(tmp_dir)
    fake_template_file = tmp_dir / "template.yaml"
    fake_template_file.write_text("fake")
    mock_argv = ["otx train", str(fake_template_file), "params", "--left-args"]
    mocker.patch("sys.argv", mock_argv)
    mock_template = mocker.patch.object(target_package, "find_and_parse_model_template")

    # run
    parser, hyper_parameters, params = get_parser_and_hprams_data()

    # check
    mock_template.assert_called_once()
    assert hyper_parameters == {}
    assert params == ["params", "--left-args"]
    assert isinstance(parser, ArgumentParser)


@e2e_pytest_unit
def test_get_parser_and_hprams_data(mocker):
    # prepare
    mock_argv = ["otx train", "params", "--left-args"]
    mocker.patch("sys.argv", mock_argv)

    # run
    parser, hyper_parameters, params = get_parser_and_hprams_data()

    # check
    assert hyper_parameters == {}
    assert params == ["params", "--left-args"]
    assert isinstance(parser, ArgumentParser)


@pytest.fixture
def fxt_argparse():
    parser = ArgumentParser()
    parser.add_argument(
        "--mem-cache-size",
        dest="params.algo_backend.mem_cache_size",
        action=MemSizeAction,
        type=str,
        required=False,
        default=0,
    )
    return parser


@pytest.mark.parametrize(
    "mem_size_arg,expected",
    [
        ("1561", 1561),
        ("121k", 121 * (2**10)),
        ("121kb", 121 * (10**3)),
        ("121kib", 121 * (2**10)),
        ("121m", 121 * (2**20)),
        ("121mb", 121 * (10**6)),
        ("121mib", 121 * (2**20)),
        ("121g", 121 * (2**30)),
        ("121gb", 121 * (10**9)),
        ("121gib", 121 * (2**30)),
        ("121as", None),
        ("121dddd", None),
    ],
)
def test_mem_size_parsing(fxt_argparse, mem_size_arg, expected):
    try:
        args = fxt_argparse.parse_args(["--mem-cache-size", mem_size_arg])
        assert getattr(args, "params.algo_backend.mem_cache_size") == expected
    except ValueError:
        assert expected is None
