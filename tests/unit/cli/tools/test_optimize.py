import argparse

import pytest

from otx.cli.tools import optimize as target_package
from otx.cli.tools.optimize import get_args, main
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_args(mocker):
    mock_options = {
        "--train-data-roots": "train_data_roots_path",
        "--val-data-roots": "val_data_roots_path",
        "--load-weights": "load_weights_path",
        "--output": "output",
        "--workspace": "work_dir_path",
    }
    mock_command = ["otx"]
    for key, value in mock_options.items():
        mock_command.extend([key, value])

    mocker.patch("sys.argv", mock_command)
    mocker.patch.object(
        target_package, "get_parser_and_hprams_data", return_value=[argparse.ArgumentParser(), {"param": "test"}, []]
    )
    mocker.patch.object(target_package, "add_hyper_parameters_sub_parser", return_value=argparse.ArgumentParser())

    parsed_args, _ = get_args()

    assert parsed_args.train_data_roots == "train_data_roots_path"
    assert parsed_args.val_data_roots == "val_data_roots_path"
    assert parsed_args.load_weights == "load_weights_path"
    assert parsed_args.output == "output"
    assert parsed_args.workspace == "work_dir_path"


@pytest.fixture
def mock_args(mocker, tmp_path):
    mock_args = mocker.MagicMock()
    mock_args.train_data_roots = "fake_train_data_roots_path"
    mock_args.val_data_roots = "fake_val_data_roots_path"
    mock_args.load_weights = "fake_load_weights_path"
    mock_args.output = tmp_path / "save/model"
    mock_args.workspace = tmp_path / "work_dir_path"

    def mock_contains(self, val):
        return val in self.__dict__

    mock_args.__contains__ = mock_contains
    mock_get_args = mocker.patch("otx.cli.tools.optimize.get_args")
    mock_get_args.return_value = [mock_args, []]

    return mock_args


@pytest.fixture
def mock_config_manager(mocker):
    mock_config_manager = mocker.patch.object(target_package, "ConfigManager")
    mock_template = mocker.MagicMock()
    mock_template.name = "fake_template_name"
    mock_config_manager.return_value.template = mock_template
    mock_config_manager.return_value.check_workspace.return_value = True
    mock_config_manager.return_value.get_dataset_config.return_value = {}
    mock_config_manager.return_value.get_hyparams_config.return_value = {}

    return mock_config_manager


@pytest.fixture
def mock_dataset_adapter(mocker):
    mock_dataset_adapter = mocker.patch("otx.cli.tools.optimize.get_dataset_adapter")
    mock_dataset = mocker.MagicMock()
    mock_label_schema = mocker.MagicMock()
    mock_dataset_adapter.return_value.get_otx_dataset.return_value = mock_dataset
    mock_dataset_adapter.return_value.get_label_schema.return_value = mock_label_schema

    return mock_dataset_adapter


@pytest.fixture
def mock_task_class(mocker):
    return mocker.patch.object(target_package, "get_impl_class")


@pytest.fixture
def mock_task(mocker, mock_task_class, mock_dataset_adapter):
    mock_task_class.return_value.return_value = mocker.MagicMock()
    mocker.patch.object(target_package, "get_dataset_adapter", return_value=mock_dataset_adapter)


@e2e_pytest_unit
def test_main(
    mocker,
    mock_args,
    mock_config_manager,
    mock_dataset_adapter,
    mock_task,
):
    mocker.patch.object(target_package, "read_model", return_value=mocker.MagicMock())

    mocker.patch("otx.cli.tools.optimize.save_model_data")

    mocker.patch.object(
        target_package,
        "ResultSetEntity",
        return_value=mocker.MagicMock(),
    )

    mocker.patch.object(
        target_package,
        "InferenceParameters",
        return_value=mocker.MagicMock(),
    )

    mocker.patch.object(
        target_package,
        "Subset",
        return_value=mocker.MagicMock(),
    )

    mocker.patch.object(
        target_package,
        "TaskEnvironment",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("json.dump")
    mocker.patch("builtins.open")

    mock_get_args = mocker.patch("otx.cli.tools.optimize.get_args")
    mock_get_args.return_value = [mock_args, []]

    ret = main()
    assert ret["retcode"] == 0
