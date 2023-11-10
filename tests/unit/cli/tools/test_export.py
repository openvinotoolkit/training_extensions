import argparse
from pathlib import Path

import pytest

from otx.cli.tools import export as target_package
from otx.cli.tools.export import get_args, main
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_args(mocker):
    mocker.patch("sys.argv", ["otx", "--load-weights", "load_weights", "--output", "output"])
    mocker.patch.object(
        target_package,
        "get_parser_and_hprams_data",
        return_value=[
            argparse.ArgumentParser(),
            {"result_based_confidence": False, "confidence_threshold": 0.35},
            [
                "params",
                "--postprocessing.result_based_confidence",
                "false",
                "--postprocessing.confidence_threshold",
                "0.95",
            ],
        ],
    )

    parsed_args, override_param = get_args()

    assert parsed_args.load_weights == "load_weights"
    assert parsed_args.output == "output"
    assert override_param == [
        "params.postprocessing.result_based_confidence",
        "params.postprocessing.confidence_threshold",
    ]


@pytest.fixture
def mock_args(mocker, tmp_dir):
    mock_args = mocker.MagicMock()
    mock_args.load_weights = "fake.bin"
    mock_args.output = tmp_dir
    mock_args.export_type = "openvino"

    def mock_contains(self, val):
        return val in self.__dict__

    mock_args.__contains__ = mock_contains
    mock_get_args = mocker.patch("otx.cli.tools.export.get_args")
    mock_get_args.return_value = (mock_args, [])

    return mock_args


@pytest.fixture
def mock_task(mocker):
    mock_task_class = mocker.MagicMock()
    mock_task = mocker.MagicMock()
    mock_task_class.return_value = mock_task
    mocker.patch.object(target_package, "get_impl_class", return_value=mock_task_class)

    return mock_task


@pytest.fixture
def mock_config_manager(mocker):
    mock_config_manager = mocker.patch.object(target_package, "ConfigManager")
    mock_template = mocker.MagicMock()
    mock_template.name = "fake_name"
    mock_config_manager.return_value.template = mock_template

    return mock_config_manager


@e2e_pytest_unit
def test_main(mocker, mock_args, mock_task, mock_config_manager, tmp_dir):
    mocker.patch.object(target_package, "is_checkpoint_nncf", return_value=True)
    mocker.patch.object(target_package, "TaskEnvironment")
    mocker.patch.object(target_package, "read_label_schema")
    mocker.patch.object(target_package, "read_binary")

    def mock_export_side_effect(export_type, output_model, precision, dump_features):
        output_model.set_data("fake.xml", b"fake")

    mock_task.export.side_effect = mock_export_side_effect
    tmp_dir = Path(tmp_dir)

    # run
    ret = main()

    # check
    assert ret["retcode"] == 0
    assert ret["template"] == "fake_name"
    mock_task.export.assert_called_once()
    with (tmp_dir / "fake.xml").open() as f:
        assert f.readline() == "fake"
