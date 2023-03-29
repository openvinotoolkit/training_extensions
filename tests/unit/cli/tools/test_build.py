import argparse

import pytest

from otx.cli.tools import build as target_package
from otx.cli.tools.build import get_args, main
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_args(mocker):
    mock_options = {
        "--train-data-roots": "train/data/root",
        "--val-data-roots": "val/data/root",
        "--test-data-roots": "test/data/root",
        "--unlabeled-data-roots": "unlabeled/data/root",
        "--unlabeled-file-list": "unlabeled/file/list",
        "--task": "detection",
        "--train-type": "Semisupervised",
        "--workspace": "work/dir/path",
        "--model": "SSD",
        "--backbone": "torchvision.resnet18",
    }
    mock_command = ["otx"]
    for key, value in mock_options.items():
        mock_command.extend([key, value])

    mocker.patch("sys.argv", mock_command)
    mocker.patch.object(target_package, "get_parser_and_hprams_data", return_value=[argparse.ArgumentParser(), {}, []])

    parsed_args = get_args()

    assert parsed_args.train_data_roots == "train/data/root"
    assert parsed_args.val_data_roots == "val/data/root"
    assert parsed_args.test_data_roots == "test/data/root"
    assert parsed_args.unlabeled_data_roots == "unlabeled/data/root"
    assert parsed_args.unlabeled_file_list == "unlabeled/file/list"
    assert parsed_args.workspace == "work/dir/path"
    assert parsed_args.task == "detection"
    assert parsed_args.train_type == "Semisupervised"
    assert parsed_args.model == "SSD"
    assert parsed_args.backbone == "torchvision.resnet18"


@pytest.fixture
def mock_args(mocker, tmp_path):
    mock_args = mocker.MagicMock()
    mock_args.train_data_roots = None
    mock_args.val_data_roots = None
    mock_args.test_data_roots = None
    mock_args.unlabeled_data_roots = None
    mock_args.unlabeled_file_list = None
    mock_args.task = ""
    mock_args.train_type = "incremental"
    mock_args.workspace = tmp_path / "work_dir"
    mock_args.model = ""
    mock_args.backbone = "torchvision.resnet18"

    def mock_contains(self, val):
        return val in self.__dict__

    mock_args.__contains__ = mock_contains
    mock_get_args = mocker.patch("otx.cli.tools.build.get_args")
    mock_get_args.return_value = mock_args

    return mock_args


@pytest.fixture
def mock_config_manager(mocker):
    mock_config_manager = mocker.patch.object(target_package, "ConfigManager")
    mock_template = mocker.MagicMock()
    mock_template.name = "fake_name"
    mock_config_manager.return_value.mode = "build"
    mock_config_manager.return_value.template = mock_template
    mock_config_manager.return_value.check_workspace.return_value = True
    mock_config_manager.return_value.get_dataset_config.return_value = {}
    mock_config_manager.return_value.get_hyparams_config.return_value = {}

    return mock_config_manager


@e2e_pytest_unit
def test_main(mocker, mock_config_manager, mock_args):
    # Mock argparse namespace
    mock_builder = mocker.patch("otx.cli.builder.Builder")

    # Call main function
    result = main()

    # Check return value
    assert result == {"retcode": 0, "task_type": ""}

    # Check ConfigManager constructor call
    mock_config_manager.assert_called_once()

    # Check ConfigManager method calls
    mock_config_manager.return_value.configure_template.assert_called_once_with(model="")
    mock_config_manager.return_value.build_workspace.assert_called_once()
    mock_config_manager.return_value.configure_data_config.assert_called_once_with()

    # Check Builder constructor call
    mock_builder.assert_called_once_with()
