import argparse
from pathlib import Path

import pytest

from otx.cli.tools import deploy as target_package
from otx.cli.tools.deploy import get_args, main
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_args(mocker):
    mocker.patch("sys.argv", ["otx", "--load-weights", "load_weights", "--output", "output"])
    mocker.patch.object(
        target_package, "get_parser_and_hprams_data", return_value=[argparse.ArgumentParser(), "fake", "fake"]
    )

    parsed_args = get_args()

    assert parsed_args.load_weights == "load_weights"
    assert parsed_args.output == "output"


@pytest.fixture
def mock_args(mocker, tmp_dir):
    mock_args = mocker.MagicMock()
    mock_args.load_weights = "fake.bin"
    mock_args.output = tmp_dir

    def mock_contains(self, val):
        return val in self.__dict__

    mock_args.__contains__ = mock_contains
    mock_get_args = mocker.patch("otx.cli.tools.deploy.get_args")
    mock_get_args.return_value = mock_args

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
    # prepare
    mocker.patch.object(target_package, "TaskEnvironment")
    mocker.patch.object(target_package, "create")
    mocker.patch.object(target_package, "read_model")
    mocker.patch.object(target_package, "read_label_schema")
    mock_deployed_model = mocker.MagicMock()
    mock_deployed_model.exportable_code = b"exportable_code"
    mocker.patch.object(target_package, "ModelEntity", return_value=mock_deployed_model)

    # run
    ret = main()

    # check
    assert ret["retcode"] == 0
    assert ret["template"] == "fake_name"
    mock_task.deploy.assert_called_once_with(mock_deployed_model)
    with (Path(tmp_dir) / "openvino.zip").open() as f:
        val = f.readline()
        assert val == "exportable_code"


@e2e_pytest_unit
def test_main_wrong_workspace(mock_args, mock_config_manager):
    mock_args.load_weights = ""
    mock_config_manager.check_workspace.return_value = True

    with pytest.raises(RuntimeError):
        main()


@e2e_pytest_unit
@pytest.mark.parametrize("load_weights", ["fake.jpg", "fake.png"])
def test_main_wrong_laod_weight(mocker, load_weights):
    # prepare
    mock_agrs = mocker.MagicMock()
    mock_agrs.load_weights = load_weights
    mocker.patch.object(target_package, "get_args", return_value=mock_agrs)
    mocker.patch.object(target_package, "ConfigManager")

    # run
    with pytest.raises(RuntimeError):
        main()
